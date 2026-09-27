# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO-like algorithms.
"""

__all__ = ["register_adv_est", "get_adv_estimator_fn", "AdvantageEstimator"]

from collections import defaultdict
from enum import Enum
from typing import Any, Callable, Optional
import re
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

import verl.utils.torch_functional as verl_F
from verl.trainer.config import AlgoConfig
from verl.utils import as_torch_index, group_mean_std
from verl.utils.import_utils import deprecated
from verl.workers.config import ActorConfig

PolicyLossFn = Callable[
    [
        torch.Tensor,  # old_log_prob
        torch.Tensor,  # log_prob
        torch.Tensor,  # advantages
        torch.Tensor,  # response_mask
        str,  # loss_agg_mode
        Optional[DictConfig | ActorConfig],  # config
        torch.Tensor | None,  # rollout_log_probs
    ],
    tuple[torch.Tensor, dict[str, Any]],
]

POLICY_LOSS_REGISTRY: dict[str, PolicyLossFn] = {}


def register_policy_loss(name: str) -> Callable[[PolicyLossFn], PolicyLossFn]:
    """Register a policy loss function with the given name.

    Args:
        name (str): The name to register the policy loss function under.

    Returns:
        function: Decorator function that registers the policy loss function.
    """

    def decorator(func: PolicyLossFn) -> PolicyLossFn:
        POLICY_LOSS_REGISTRY[name] = func
        return func

    return decorator


def get_policy_loss_fn(name):
    """Get the policy loss with a given name.

    Args:
        name: `(str)`
            The name of the policy loss.

    Returns:
        `(callable)`: The policy loss function.
    """
    loss_name = name
    if loss_name not in POLICY_LOSS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(POLICY_LOSS_REGISTRY.keys())}"
        )
    return POLICY_LOSS_REGISTRY[loss_name]


class AdvantageEstimator(str, Enum):
    """Using an enumeration class to avoid spelling errors in adv_estimator.

    Note(haibin.lin): this enum class is immutable after creation. Extending this
    enum for new estimators may not be necessary since users can always just call
    `verl.trainer.ppo.core_algos.register` with string name for a custom advantage
    estimator instead.
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    OPO = "opo"
    GRPO_PASSK = "grpo_passk"
    GPG = "gpg"
    RLOO_VECTORIZED = "rloo_vectorized"
    GRPO_VECTORIZED = "grpo_vectorized"
    OPTIMAL_TOKEN_BASELINE = "optimal_token_baseline"
    TIR_OPTIMAL_TOKEN_BASELINE = "tir_optimal_token_baseline"


ADV_ESTIMATOR_REGISTRY: dict[str, Any] = {}


def register_adv_est(name_or_enum: str | AdvantageEstimator) -> Any:
    """Decorator to register a advantage estimator function with a given name.

    Args:
        name_or_enum: `(str)` or `(AdvantageEstimator)`
            The name or enum of the advantage estimator.

    """

    def decorator(fn):
        name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
        if name in ADV_ESTIMATOR_REGISTRY and ADV_ESTIMATOR_REGISTRY[name] != fn:
            raise ValueError(
                f"Adv estimator {name} has already been registered: {ADV_ESTIMATOR_REGISTRY[name]} vs {fn}"
            )
        ADV_ESTIMATOR_REGISTRY[name] = fn
        return fn

    return decorator


def get_adv_estimator_fn(name_or_enum):
    """Get the advantage estimator function with a given name.

    Args:
        name_or_enum: `(str)` or `(AdvantageEstimator)`
            The name or enum of the advantage estimator.

    Returns:
        `(callable)`: The advantage estimator function.
    """
    name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
    if name not in ADV_ESTIMATOR_REGISTRY:
        raise ValueError(f"Unknown advantage estimator simply: {name}")
    return ADV_ESTIMATOR_REGISTRY[name]


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        """Update the KL coefficient based on current KL divergence.

        Args:
            current_kl (float): Current KL divergence value.
            n_steps (int): Number of steps taken.
        """
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        """Update method for fixed KL controller (no-op).

        Args:
            current_kl (float): Current KL divergence value (unused).
            n_steps (int): Number of steps taken (unused).
        """
        pass


def get_kl_controller(kl_ctrl):
    """Factory function to create appropriate KL controller based on configuration.

    Args:
        kl_ctrl: Configuration object containing KL controller settings.

    Returns:
        KL controller instance (FixedKLController or AdaptiveKLController).

    Raises:
        NotImplementedError: If controller type is not supported.
        AssertionError: If adaptive controller horizon is not positive.
    """
    if kl_ctrl.type == "fixed":
        return FixedKLController(kl_coef=kl_ctrl.kl_coef)
    elif kl_ctrl.type == "adaptive":
        assert kl_ctrl.horizon > 0, f"horizon must be larger than 0. Got {kl_ctrl.horizon}"
        return AdaptiveKLController(init_kl_coef=kl_ctrl.kl_coef, target_kl=kl_ctrl.target_kl, horizon=kl_ctrl.horizon)
    else:
        raise NotImplementedError


@register_adv_est(AdvantageEstimator.GAE)  # or simply: @register_adv_est("gae")
def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        values: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma is `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        nextvalues = 0
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam_ = delta + gamma * lam * lastgaelam

            # skip values and TD-error on observation tokens
            nextvalues = values[:, t] * response_mask[:, t] + (1 - response_mask[:, t]) * nextvalues
            lastgaelam = lastgaelam_ * response_mask[:, t] + (1 - response_mask[:, t]) * lastgaelam

            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, response_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
@register_adv_est(AdvantageEstimator.GRPO)  # or simply: @register_adv_est("grpo")
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length)
        index: `(np.ndarray)`
            index array for grouping
        epsilon: `(float)`
            small value to avoid division by zero
        norm_adv_by_std_in_grpo: `(bool)`
            whether to scale the GRPO advantage
        config: `(Optional[AlgoConfig])`
            algorithm configuration object

    Note:
        If norm_adv_by_std_in_grpo is True, the advantage is scaled by the std, as in the original GRPO.
        If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, response_length)
        Returns: `(torch.Tensor)`
            shape is (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]

        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.GRPO_VECTORIZED)
def compute_grpo_vectorized_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized GRPO（outcome-only）:
      For each group g:
      a_i = \\frac{r_i - \\mu_g}{\\sigma_g} (or without dividing by \\sigma_g),
      then broadcast the scalar across the token dimension (multiplied by response_mask).。
    """
    with torch.no_grad():
        scores = token_level_rewards.sum(dim=-1)
        g = as_torch_index(index, device=scores.device)
        mean_g, std_g, _ = group_mean_std(scores, g, eps=epsilon)
        if norm_adv_by_std_in_grpo:
            scalars = (scores - mean_g[g]) / (std_g[g] + epsilon)
        else:
            scalars = scores - mean_g[g]

        advantages = scalars.unsqueeze(-1) * response_mask
        return advantages, advantages


@register_adv_est(AdvantageEstimator.GRPO_PASSK)  # or simply: @register_adv_est("grpo_passk")
def compute_grpo_passk_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for Pass@k using a GRPO-style outcome reward formulation.
    Only the best response per group gets a non-zero advantage: r_max - r_second_max.

    Implemented as described in https://arxiv.org/abs/2503.19595.

    Args:
        token_level_rewards: (bs, response_length)
        response_mask: (bs, response_length)
        index: (bs,) → group ID per sample
        epsilon: float for numerical stability
        config: (AlgoConfig) algorithm settings, which contains "norm_adv_by_std_in_grpo"

    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length)
    """
    assert config is not None
    # if True, normalize advantage by std within group
    norm_adv_by_std_in_grpo = config.get("norm_adv_by_std_in_grpo", True)
    scores = token_level_rewards.sum(dim=-1)  # (bs,)
    advantages = torch.zeros_like(scores)

    id2scores = defaultdict(list)
    id2indices = defaultdict(list)

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            idx = index[i]
            id2scores[idx].append(scores[i])
            id2indices[idx].append(i)

        for idx in id2scores:
            rewards = torch.stack(id2scores[idx])  # (k,)
            if rewards.numel() < 2:
                raise ValueError(
                    f"Pass@k requires at least 2 samples per group. Got {rewards.numel()} for group {idx}."
                )
            topk, topk_idx = torch.topk(rewards, 2)
            r_max, r_second_max = topk[0], topk[1]
            i_max = id2indices[idx][topk_idx[0].item()]
            advantage = r_max - r_second_max
            if norm_adv_by_std_in_grpo:
                std = torch.std(rewards)
                advantage = advantage / (std + epsilon)
            advantages[i_max] = advantage

    advantages = advantages.unsqueeze(-1) * response_mask
    return advantages, advantages


@register_adv_est(
    AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE
)  # or simply: @register_adv_est("reinforce_plus_plus_baseline")
def compute_reinforce_plus_plus_baseline_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: torch.Tensor,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RF++-baseline (https://arxiv.org/abs/2501.03262), operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]

        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        scores = verl_F.masked_whiten(scores, response_mask) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.RLOO)  # or simply: @register_adv_est("rloo")
def compute_rloo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num - 1) - id2mean[index[i]] * response_num / (
                    response_num - 1
                )
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.OPO)  # or simply: @register_adv_est("opo")
def compute_opo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for OPO based on https://arxiv.org/pdf/2505.23585

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = response_mask.sum(dim=-1)
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2len = defaultdict(list)
    id2bsl = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
            id2len[index[i]].append(response_length[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2bsl[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                score_tensor = torch.stack(id2score[idx])
                len_tensor = torch.stack(id2len[idx])
                id2bsl[idx] = (len_tensor * score_tensor).sum() / len_tensor.sum()
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2bsl[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.REINFORCE_PLUS_PLUS)  # or simply: @register_adv_est("reinforce_plus_plus")
def compute_reinforce_plus_plus_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, config: Optional[AlgoConfig] = None, **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    assert config is not None
    gamma = config.gamma
    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * response_mask[:, t]

        advantages = verl_F.masked_whiten(returns, response_mask)
        advantages = advantages * response_mask

    return advantages, returns


@register_adv_est(AdvantageEstimator.REMAX)  # or simply: @register_adv_est("remax")
def compute_remax_outcome_advantage(
    token_level_rewards: torch.Tensor,
    reward_baselines: torch.Tensor,
    response_mask: torch.Tensor,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = returns - reward_baselines.unsqueeze(-1) * response_mask

    return advantages, returns


@register_adv_est(AdvantageEstimator.GPG)  # or simply: @register_adv_est("gpg")
def compute_gpg_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    f_norm: float = 1.0,
    alpha: float = 1.0,
    config=None,
    **kwargs,
):
    """
    Compute advantage for GPG, operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        index: `(np.ndarray)`
            shape: (bs,)
        epsilon: (float)
        f_norm: (float)
        alpha: (float)
        config: (dict) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        m = torch.count_nonzero(scores)
        alpha = bsz / m.clamp(min=1)

        for i in range(bsz):
            id2score[index[i]].append(scores[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = alpha * (scores[i] - id2mean[index[i]]) / (f_norm)
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.RLOO_VECTORIZED)  # or simply: @register_adv_est("rloo_vectorized")
def compute_rloo_vectorized_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    with torch.no_grad():
        inv = torch.from_numpy(np.unique(index, return_inverse=True)[1]).to(scores.device)

        c = torch.bincount(inv)[inv].to(scores.dtype)
        adv = ((c * scores - torch.bincount(inv, weights=scores)[inv]) / (c - 1).clamp_min(1)) * (c > 1)

        adv = adv.unsqueeze(-1) * response_mask

    return adv, adv


@register_adv_est(AdvantageEstimator.OPTIMAL_TOKEN_BASELINE)
def compute_optimal_token_baseline_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    old_log_probs: torch.Tensor,
    sum_pi_squared: torch.Tensor,
    rollout_is_weights: torch.Tensor = None,
    handle_zero_tail: bool = False,
    epsilon: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantages using Optimal Token Baseline (OTB).

    Unlike the group mean based baseline which uses a single baseline per trajectory,
    this computes a unique baseline for each timestep using cumulative path variance.

    Theory:
        For each timestep t in each prompt group:
            B_t* = E[G_t × W_t] / E[W_t]
        where W_t = Σ_{j=1}^t ||s_j||² (cumulative path-variance proxy)
        and ||s_j||² = 1 - 2π_j + Σπ²

    The cumulative sum W_t captures the "realized energy" of trajectory has been up to timestep t,
    giving higher weight to predicting rewards on high-variance paths.

    Args:
        token_level_rewards: Rewards at each token position [shape: (bs, response_length)]
        response_mask: Binary mask for valid tokens (1) vs padding (0) [shape: (bs, response_length)]
        index: Prompt indices for grouping trajectories from same prompt [shape: (bs,)]
        old_log_probs: Log probabilities from training policy during generation [shape: (bs, response_length)]
        sum_pi_squared: Sum of squared probabilities over vocabulary Σπ² [shape: (bs, response_length)]
        rollout_is_weights: Pre-computed IS weights for W correction [shape: (bs, response_length)],
            None if not using IS
        handle_zero_tail: If True, zero baselines will be set in the portion of the longest trajectory
            that extends beyond the second-longest trajectory in the prompt group.
            Default: False
        epsilon: Small constant for numerical stability (default: 1e-8)

    Returns:
        advantages: OTB advantage estimates [shape: (bs, response_length)]
        returns: Cumulative rewards (returns) from each position [shape: (bs, response_length)]

    Note on Rollout Importance Sampling:
        When rollout_is_weights is provided, W_t is scaled by ρ̄²(t) to minimize MSE under truncated IS:
            B_t* = Σ[G_t × ρ̄²(t) × W_t] / Σ[ρ̄²(t) × W_t]
    """
    with torch.no_grad():
        batch_size, seq_len = token_level_rewards.shape
        device = token_level_rewards.device

        # Compute returns (reward-to-go) for each timestep
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])

        # Step 1: Compute w_per_timestep = 1 - 2π_t + Σπ²)
        pi_t = torch.exp(old_log_probs)
        w_per_timestep = 1 - 2 * pi_t + sum_pi_squared

        # Step 2: Apply rollout importance sampling correction (if enabled)
        if rollout_is_weights is not None:
            # Scale W by ρ̄² to minimize MSE under truncated IS
            w_per_timestep = w_per_timestep * (rollout_is_weights**2)

        # Step 3: Compute cumulative path-variance proxy: W_t = Σ_{j=1}^t w_j
        # This measures accumulated variance from the start of the trajectory up to timestep t
        w_cumulative = (w_per_timestep * response_mask).cumsum(dim=-1)

        # Group trajectories by prompt
        prompt_groups = defaultdict(list)
        for i in range(batch_size):
            prompt_groups[index[i]].append(i)

        # Initialize baselines tensor [batch_size, seq_len]
        baselines = torch.zeros_like(returns)

        # Compute per-step baseline for each prompt group
        for _, trajectory_indices in prompt_groups.items():
            N = len(trajectory_indices)
            if N == 1:
                # Single trajectory - no baseline (advantage = return)
                continue

            traj_idx = torch.tensor(trajectory_indices, device=device)

            # Extract group data [N, seq_len]
            returns_group = returns[traj_idx]
            w_cumulative_group = w_cumulative[traj_idx]
            mask_group = response_mask[traj_idx]

            # Compute per-timestep baseline: B_t = Σ[G_t × W_t] / Σ[W_t]
            # where W_t = Σ_{j=1}^t ||s_j||² (cumulative path variance)
            # Shape: [seq_len]
            numerator = (returns_group * w_cumulative_group * mask_group).sum(dim=0)  # Sum over trajectories
            denominator = (w_cumulative_group * mask_group).sum(dim=0) + epsilon

            baseline_per_step = numerator / denominator  # [seq_len]

            # Assign to all trajectories in this group
            baselines[traj_idx] = baseline_per_step.unsqueeze(0).expand(N, -1)

            if handle_zero_tail:
                # Optionally zero out the portion of the longest trajectory that extends
                # beyond the second-longest trajectory in the prompt group.
                response_lengths = mask_group.sum(dim=-1)
                sorted_lengths, _ = torch.sort(response_lengths)
                max_length = int(sorted_lengths[-1].item())
                second_max_length = int(sorted_lengths[-2].item())
                max_length_idx = (response_lengths == max_length).nonzero(as_tuple=True)[0]
                if max_length_idx.numel() == 1 and max_length > second_max_length:
                    max_length_traj_idx = trajectory_indices[int(max_length_idx[0])]
                    baselines[max_length_traj_idx, second_max_length:] = 0.0

        # Compute advantages: A_t = G_t - B_t
        advantages = (returns - baselines) * response_mask

    return advantages, returns


@register_adv_est(AdvantageEstimator.TIR_OPTIMAL_TOKEN_BASELINE)
def compute_multi_turn_optimal_token_baseline_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    old_log_probs: torch.Tensor,
    sum_pi_squared: torch.Tensor,
    rollout_is_weights: torch.Tensor = None,
    handle_zero_tail: bool = True,
    epsilon: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantages using Optimal Token Baseline (OTB).

    Unlike the group mean based baseline which uses a single baseline per trajectory,
    this computes a unique baseline for each timestep using cumulative path variance.

    Theory:
        For each timestep t in each prompt group:
            B_t* = E[G_t × W_t] / E[W_t]
        where W_t = Σ_{j=1}^t ||s_j||² (cumulative path-variance proxy)
        and ||s_j||² = 1 - 2π_j + Σπ²

    The cumulative sum W_t captures the "realized energy" of trajectory has been up to timestep t,
    giving higher weight to predicting rewards on high-variance paths.

    Args:
        token_level_rewards: Rewards at each token position [shape: (bs, response_length)]
        response_mask: Binary mask for valid tokens (1) vs padding (0) [shape: (bs, response_length)]
        index: Prompt indices for grouping trajectories from same prompt [shape: (bs,)]
        old_log_probs: Log probabilities from training policy during generation [shape: (bs, response_length)]
        sum_pi_squared: Sum of squared probabilities over vocabulary Σπ² [shape: (bs, response_length)]
        rollout_is_weights: Pre-computed IS weights for W correction [shape: (bs, response_length)],
            None if not using IS
        handle_zero_tail: If True, zero baselines will be set in the portion of the longest trajectory
            that extends beyond the second-longest trajectory in the prompt group.
            Default: False
        epsilon: Small constant for numerical stability (default: 1e-8)

    Returns:
        advantages: OTB advantage estimates [shape: (bs, response_length)]
        returns: Cumulative rewards (returns) from each position [shape: (bs, response_length)]

    Note on Rollout Importance Sampling:
        When rollout_is_weights is provided, W_t is scaled by ρ̄²(t) to minimize MSE under truncated IS:
            B_t* = Σ[G_t × ρ̄²(t) × W_t] / Σ[ρ̄²(t) × W_t]
    """
    with torch.no_grad():
        # Compute returns (reward-to-go) for each timestep
        token_returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])

        # Step 1: Compute w_per_timestep = 1 - 2π_t + Σπ²)
        pi_t = torch.exp(old_log_probs)
        w_per_timestep = 1 - 2 * pi_t + sum_pi_squared

        # Step 2: Apply rollout importance sampling correction (if enabled)
        if rollout_is_weights is not None:
            # Scale W by ρ̄² to minimize MSE under truncated IS
            w_per_timestep = w_per_timestep * (rollout_is_weights**2)

        # Step 3: Compute cumulative path-variance proxy: W_t = Σ_{j=1}^t w_j
        # This measures accumulated variance from the start of the trajectory up to timestep t
        w_cumulative = (w_per_timestep * response_mask).cumsum(dim=-1)

        # Step 4: Concatenate returns and w_cumulative for each trajectory
        # This allows us to compute baseline per timestep for each trajectory
        response_lengths = response_mask.sum(dim=-1).to(dtype=torch.long)  # [shape: (bs * n, )]
        max_response_length = int(response_lengths.max().item()) if response_lengths.numel() > 0 else 0
        all_w_values = w_cumulative.new_zeros(
            (len(response_lengths), max_response_length)
        )  # [shape: (bs * n, max_response_length)]
        all_returns = torch.zeros_like(all_w_values)
        for i in range(len(response_lengths)):
            length = int(response_lengths[i].item())
            if length == 0:
                continue
            mask = response_mask[i].bool()
            all_w_values[i, :length] = w_cumulative[i, mask]
            all_returns[i, :length] = token_returns[i, mask]

        # Group trajectories by prompt
        prompt_groups = defaultdict(list)
        for i in range(len(response_lengths)):
            if response_lengths[i] == 0:
                continue
            prompt_groups[index[i]].append(i)

        # Compute optimal baseline for each prompt group
        baselines = torch.zeros_like(all_returns)

        for _, trajectory_indices in prompt_groups.items():
            N = len(trajectory_indices)
            traj_idx = torch.tensor(trajectory_indices, device=all_returns.device)

            if N == 1:
                # Single trajectory - no baseline (keep original reward as advantage)
                baselines[traj_idx[0]] = 0.0
                continue

            # Extract group data
            w_group = all_w_values[traj_idx]  # [shape: (N, max_response_length)]
            R_group = all_returns[traj_idx]  # [shape: (N, max_response_length)]
            # Direct optimal baseline - single value for all in group
            b_star = (R_group * w_group).sum(dim=0) / (w_group.sum(dim=0) + epsilon)
            # Convert to match baselines dtype (epsilon can cause float64 promotion)
            baselines[traj_idx] = b_star.to(baselines.dtype)

            if handle_zero_tail:
                # Optionally zero out the portion of the longest trajectory that extends
                # beyond the second-longest trajectory in the prompt group.
                response_lengths_group = response_lengths[traj_idx]
                sorted_lengths, _ = torch.sort(response_lengths_group)
                max_length = int(sorted_lengths[-1].item())
                second_max_length = int(sorted_lengths[-2].item())
                max_length_idx = (response_lengths_group == max_length).nonzero(as_tuple=True)[0]
                if max_length_idx.numel() == 1 and max_length > second_max_length:
                    max_length_traj_idx = trajectory_indices[int(max_length_idx[0])]
                    baselines[max_length_traj_idx, second_max_length:] = 0.0

        # Compute advantages
        all_advantages = all_returns - baselines  # [shape: (bs * n, max_response_length)]

        advantages = torch.zeros_like(token_returns)  # [shape: (bs * n, turn * response_length)]
        for i in range(len(response_lengths)):
            if response_lengths[i] == 0:
                continue
            advantages[i, response_mask[i].bool()] = all_advantages[i, : response_lengths[i]]

        advantages = advantages * response_mask  # [shape: (bs * n * turn, response_length)]

    return advantages, token_returns


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    """Compute token-level rewards with KL penalty.

    Args:
        token_level_scores (torch.Tensor): Token-level reward scores.
        old_log_prob (torch.Tensor): Log probabilities from current policy.
        ref_log_prob (torch.Tensor): Log probabilities from reference policy.
        kl_ratio (float): KL penalty coefficient.

    Returns:
        torch.Tensor: Token-level rewards with KL penalty applied.
    """
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def agg_loss(
    loss_mat: torch.Tensor,
    loss_mask: torch.Tensor,
    loss_agg_mode: str,
    dp_size: int = 1,
    batch_num_tokens: Optional[int] = None,
    global_batch_size: Optional[int] = None,
    loss_scale_factor: Optional[int] = None,
):
    """
    Aggregate the loss across global batch to ensure the loss is invariant to fsdp/megatron parallelism.

    NOTE: The returned loss has different behaviors for different backend:
    - FSDP: the loss is directly used for backward.
    - Megatron: the loss should be scaled by `num_microbatches` and `cp_size` for pp schedule.

    Args:
        loss_mat: micro batch loss matrix, (bs, response_length)
        loss_mask: micro batch loss mask, (bs, response_length)
        loss_agg_mode: method to aggregate the loss matrix into a scalar
        dp_size: data parallel size
        batch_num_tokens: number of valid tokens in global batch
        global_batch_size: global batch size
        loss_scale_factor: scale factor for "seq-mean-token-sum-norm" mode. If None, uses loss_mask.shape[-1].
            Set this to a constant value to ensure consistent normalization throughout training.

    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        if batch_num_tokens is None:
            batch_num_tokens = loss_mask.sum()
        loss = verl_F.masked_sum(loss_mat, loss_mask) / batch_num_tokens * dp_size
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        seq_mask = (torch.sum(loss_mask, dim=-1) > 0).float()  # exclude fully masked sequences
        if global_batch_size is None:
            global_batch_size = seq_mask.sum()
        loss = verl_F.masked_sum(seq_losses, seq_mask) / global_batch_size * dp_size  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_mask = torch.sum(loss_mask, dim=-1)  # per-sequence token count
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / (seq_mask + 1e-8)  # token-mean
        seq_mask = (seq_mask > 0).float()  # exclude fully masked sequences
        if global_batch_size is None:
            global_batch_size = seq_mask.sum()
        loss = verl_F.masked_sum(seq_losses, seq_mask) / global_batch_size * dp_size  # seq-mean
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        if loss_scale_factor is None:
            loss_scale_factor = loss_mask.shape[-1]
        loss = torch.sum(seq_losses) / loss_scale_factor
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


def compute_self_distillation_loss(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    self_distillation_config: Any,
    old_log_probs: Optional[torch.Tensor] = None,
    student_all_log_probs: Optional[torch.Tensor] = None,
    teacher_all_log_probs: Optional[torch.Tensor] = None,
    student_topk_log_probs: Optional[torch.Tensor] = None,
    teacher_topk_log_probs: Optional[torch.Tensor] = None,
    self_distillation_mask: Optional[torch.Tensor] = None,
    loss_agg_mode: str = "token-mean",
    rollout_is_weights: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, Any]]:

    metrics = {}

    loss_mask = response_mask
    if self_distillation_mask is not None:
        loss_mask = loss_mask * self_distillation_mask.unsqueeze(1)

    if self_distillation_config.full_logit_distillation:
        use_topk = self_distillation_config.distillation_topk is not None
        if use_topk:
            if student_topk_log_probs is None or teacher_topk_log_probs is None:
                raise ValueError("top-k distillation requires student_topk_log_probs and teacher_topk_log_probs.")

            def add_tail(log_probs: torch.Tensor) -> torch.Tensor:
                # Compute tail log-probability using logsumexp for numerical stability
                # log(1 - sum(p_i)) = log(1 - exp(log_sum_exp(log(p_i))))
                log_s = torch.logsumexp(log_probs, dim=-1, keepdim=True)
                log_s = torch.clamp(log_s, max=-1e-7)  # Clamp to avoid log_s >= 0 (which implies sum(probs) >= 1)
                tail_log = torch.log(-torch.expm1(log_s))  # We use the identity: 1 - exp(x) = -(exp(x) - 1); torch.expm1(x) computes (e^x - 1) with high precision for small x.
                return torch.cat([log_probs, tail_log], dim=-1)

            def renorm_topk_log_probs(logp: torch.Tensor) -> torch.Tensor:
                logZ = torch.logsumexp(logp, dim=-1, keepdim=True)
                return logp - logZ

            student_distill_log_probs = student_topk_log_probs
            teacher_distill_log_probs = teacher_topk_log_probs.detach()
            if self_distillation_config.distillation_add_tail:
                student_distill_log_probs = add_tail(student_distill_log_probs)
                teacher_distill_log_probs = add_tail(teacher_distill_log_probs)
            else:
                student_distill_log_probs = renorm_topk_log_probs(student_distill_log_probs)
                teacher_distill_log_probs = renorm_topk_log_probs(teacher_distill_log_probs)
        else:
            if student_all_log_probs is None or teacher_all_log_probs is None:
                raise ValueError("full_logit_distillation requires student_all_log_probs and teacher_all_log_probs.")
            student_distill_log_probs = student_all_log_probs
            teacher_distill_log_probs = teacher_all_log_probs.detach()

        if self_distillation_config.alpha == 0.0:
            kl_loss = F.kl_div(
                student_distill_log_probs, teacher_distill_log_probs, reduction="none", log_target=True
            )
        elif self_distillation_config.alpha == 1.0:
            kl_loss = F.kl_div(
                teacher_distill_log_probs, student_distill_log_probs, reduction="none", log_target=True
            )
        else:
            # Compute the log of the mixture distribution
            # log(a + b) = log(exp(log(a)) + exp(log(b))) -> for mixture
            alpha = torch.tensor(
                self_distillation_config.alpha,
                dtype=student_distill_log_probs.dtype,
                device=student_distill_log_probs.device,
            )
            mixture_log_probs = torch.logsumexp(
                torch.stack([student_distill_log_probs + torch.log(1 - alpha), teacher_distill_log_probs + torch.log(alpha)]),
                dim=0,
            )
            kl_teacher = F.kl_div(mixture_log_probs, teacher_distill_log_probs, reduction="none", log_target=True)
            kl_student = F.kl_div(mixture_log_probs, student_distill_log_probs, reduction="none", log_target=True)
            kl_loss = torch.lerp(kl_student, kl_teacher, alpha)  # Compute the Generalized Jensen-Shannon Divergence

        per_token_loss = kl_loss.sum(-1)
    else:
        assert self_distillation_config.alpha == 1.0, "Only reverse KL is supported for non-full-logit distillation"
        log_ratio = student_log_probs - teacher_log_probs.detach()
        per_token_loss = log_ratio.detach() * student_log_probs

    is_clip = self_distillation_config.is_clip
    if is_clip is not None:
        if old_log_probs is None:
            raise ValueError("old_log_probs is required for distillation IS ratio.")

        negative_approx_kl = (student_log_probs - old_log_probs).detach()
        negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
        ratio = torch.exp(negative_approx_kl).clamp(max=is_clip)
        per_token_loss = per_token_loss * ratio

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        per_token_loss = per_token_loss * rollout_is_weights

    loss = agg_loss(
        loss_mat=per_token_loss,
        loss_mask=loss_mask,
        loss_agg_mode=loss_agg_mode,
        batch_num_tokens=loss_mask.sum().clamp(min=1.0),
    )
    return loss, metrics


def _evidence_stats(raw: torch.Tensor, divergence_mask: torch.Tensor, prefix: str) -> dict[str, float]:
    """Per-token statistics of a teacher evidence over the tokens that carry a teacher term,
    logged as {prefix}/*: "opd" for A_d = log pi_teacher(y_t | x, r) - log pi_student(y_t | x)
    (every loss_mode="sipo" micro batch), "contrast" for e_t = log pi_T(y_t | x, gold) -
    log pi_T(y_t | x, wrong).

    A micro batch with no such token reports NaN, which reduce_metrics ignores, so each
    curve is the statistic over the tokens that were measured rather than a value scaled
    by P(micro batch active).
    """
    keys = ("mean", "rms", "p01", "p50", "p99", "min", "max",
            "mean_first_quarter", "mean_last_quarter", "head_tail_gap")
    with torch.no_grad():
        sel = raw[divergence_mask.bool()].float()
        if sel.numel() == 0:
            out = {f"{prefix}/{k}": float("nan") for k in keys}
            out.update({f"{prefix}/num_tokens": 0.0, f"{prefix}/active": 0.0})
            return out
        q = torch.quantile(sel, torch.tensor([0.01, 0.5, 0.99], device=sel.device, dtype=sel.dtype))
        # A_d by position: a teacher holding the answer tends to score early deliberation
        # below what the student gives it and the close above. tail - head > 0 is that
        # profile; the RLSD clip bounds how much of it can reach the advantage.
        mask_b = divergence_mask.bool()
        lengths = divergence_mask.sum(dim=-1, keepdim=True).clamp(min=1)
        frac_pos = (divergence_mask.cumsum(dim=-1) - 1) / lengths
        first_q = raw[mask_b & (frac_pos < 0.25)].float()
        last_q = raw[mask_b & (frac_pos >= 0.75)].float()
        head = first_q.mean().item() if first_q.numel() else float("nan")
        tail = last_q.mean().item() if last_q.numel() else float("nan")
        return {
            f"{prefix}/mean": sel.mean().item(),
            f"{prefix}/rms": sel.pow(2).mean().sqrt().item(),
            f"{prefix}/p01": q[0].item(), f"{prefix}/p50": q[1].item(), f"{prefix}/p99": q[2].item(),
            f"{prefix}/min": sel.min().item(), f"{prefix}/max": sel.max().item(),
            f"{prefix}/mean_first_quarter": head, f"{prefix}/mean_last_quarter": tail,
            f"{prefix}/head_tail_gap": tail - head,
            f"{prefix}/num_tokens": float(sel.numel()),
            f"{prefix}/active": 1.0,
        }


def extract_boxed_answer(text: Optional[str]) -> Optional[str]:
    """Content of the LAST balanced \\boxed{...} in `text`, stripped; None if absent, unbalanced
    or empty. Pure string matching, same rule as the math reward's extractor."""
    if not text:
        return None
    idx = text.rfind("\\boxed{")
    if idx < 0:
        return None
    depth, i = 0, idx
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                inner = text[idx + len("\\boxed{"): i].strip()
                return inner or None
        i += 1
    return None


_CODE_SOURCES = ("code", "livecodebench", "humanevalplus")


def tooluse_reference_answer(ground_truth) -> Optional[str]:
    """The tool-use ground truth rendered exactly as the tool-use scorer reports a rollout's
    answer (feedback/tooluse.py: 'Actions: [...], Inputs: {...}'), so the gold and the
    wrong-answer teacher prompts differ only in content, never in format. None if unparseable."""
    import json
    try:
        gt_list = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
        actions = [item["Action"] for item in gt_list]
        inputs: dict = {}
        for item in gt_list:
            raw = item.get("Action_Input", {})
            try:
                parsed = json.loads(raw) if isinstance(raw, str) else raw
            except json.JSONDecodeError:
                parsed = {}
            if parsed:
                inputs.update(parsed)
        return f"Actions: {actions}, Inputs: {inputs}"
    except (TypeError, KeyError, ValueError, AttributeError):
        return None


def render_reference_answer(data_source, ground_truth) -> Optional[str]:
    """The ground truth in the SAME format the task's scorer reports a rollout's answer ("pred"),
    so the gold-answer teacher prompt and the wrong-answer teacher prompt differ only in content.
    math (boxed content) and gpqa / sciknoweval (a letter): the string as is. tooluse: the
    scorer's 'Actions: [...], Inputs: {...}' rendering of the JSON action list. Code sources:
    None, the ground truth is a test suite, not an answer (sipo_contrast_context=trajectory)."""
    if ground_truth is None:
        return None
    ds = str(data_source) if data_source is not None else ""
    if ds in _CODE_SOURCES:
        return None
    if ds == "tooluse":
        return tooluse_reference_answer(ground_truth)
    text = str(ground_truth).strip()
    return text or None


_MCQ_SOURCES = ("gpqa", "sciknoweval")


def valid_judged_answer(data_source, pred) -> Optional[str]:
    """The scorer's "pred" if it is an answer in the task's answer format, else None.

    The multiple-choice scorers fall back to the whole response when they find no answer tag
    or option letter (feedback/mcq.py: text.split("<answer>")[-1]; feedback/gpqa.py:
    pred.strip()), so a malformed or truncated science rollout reports its entire text as
    "pred". That is not an answer: shown to the wrong-answer teacher as the "reference
    solution" it would make the two prompts differ by a whole essay and hand the teacher the
    row's own text. gpqa / sciknoweval: a single option letter. tooluse: the scorer's
    'Actions: [...], Inputs: {...}' rendering. Code sources: no answer at all. Math and the
    rest: the boxed content the verifier read (bounded by its 100-character window)."""
    if pred is None:
        return None
    p = str(pred).strip()
    if not p:
        return None
    ds = str(data_source) if data_source is not None else ""
    if ds in _MCQ_SOURCES:
        return p if re.fullmatch(r"[A-Z]", p) else None
    if ds == "tooluse":
        return p if p.startswith("Actions:") else None
    if ds in _CODE_SOURCES:
        return None
    return p


def judged_answers(preds: Optional[list], response_texts: list, data_sources: Optional[list] = None) -> tuple[list, int, int]:
    """Per row, the final answer the VERIFIER judged: the reward manager's "pred" (math: the last
    \\boxed{} within the final 100 characters of the response, feedback/math.py
    is_correct_strict_box), kept only if valid_judged_answer accepts it for the row's task;
    "" or None means the verifier saw no answer. The whole-response boxed extraction is used
    only when the reward manager reported no preds at all.

    Returns (answers, n_unjudged_box, n_invalid): rows with no judged answer whose response still
    carries a \\boxed{} earlier (boxed, then kept writing or was cut off, scored 0), and rows whose
    non-empty pred was rejected as not an answer (e.g. a science rollout with no answer tag)."""
    n = len(response_texts)
    n_invalid = 0
    if preds is not None and len(preds) == n:
        answers = []
        for i, p in enumerate(preds):
            raw = (str(p).strip() or None) if p is not None else None
            ds = data_sources[i] if data_sources is not None and i < len(data_sources) else None
            a = valid_judged_answer(ds, raw) if data_sources is not None else raw
            n_invalid += int(raw is not None and a is None)
            answers.append(a)
    else:
        answers = [extract_boxed_answer(t) for t in response_texts]
    n_unjudged = sum(1 for i in range(n) if answers[i] is None and extract_boxed_answer(response_texts[i]) is not None)
    return answers, n_unjudged, n_invalid

def group_mode_wrong_answer(answers: list, ok: list, uids, golds: Optional[list] = None) -> list:
    """Per row, the most common WRONG final answer in its group (ties: first seen), or None when
    the group has no parseable wrong answer. This is the contrast context for the second
    teacher: the plausible distractor the group actually fell for. An answer equal to the
    row's gold string is never a candidate, so a rollout the verifier failed for another
    reason (truncation, format) cannot hand the wrong-answer teacher the right answer."""
    per_group: dict = {}
    for i, (a, k, u) in enumerate(zip(answers, ok, uids)):
        if k or a is None or (golds is not None and golds[i] is not None and a.strip() == str(golds[i]).strip()):
            continue
        counts = per_group.setdefault(u, {})
        counts[a] = counts.get(a, 0) + 1
    mode = {u: max(c.items(), key=lambda kv: kv[1])[0] for u, c in per_group.items() if c}
    return [mode.get(u) for u in uids]


def own_or_group_wrong_answer(answers: list, ok: list, uids, golds: Optional[list] = None,
                              unjudged: str = "skip") -> tuple[list, list]:
    """Per row, the wrong answer the second teacher is shown.

    A WRONG rollout is contrasted against its own judged final answer, the answer it actually
    reached. A wrong rollout WITHOUT one (truncated, answer outside the verifier's window,
    malformed, or equal to the gold string) has no matched negative: with unjudged="skip" it
    gets None, i.e. no teacher term; any other wrong answer is a mismatch whose credit has no
    reliable sign (both teachers disfavour the row's own path), and on the math runs those rows
    carried a net positive push that grew with truncation. unjudged="mode" restores the earlier
    fallback to the group's most common wrong answer. A CORRECT rollout always uses the group
    mode: its question is "why gold and not a common error", which needs no match.

    Returns (answers, own_used): own_used[i] is True where the row's own answer was used."""
    if unjudged not in ("skip", "mode"):
        raise ValueError(f"unjudged must be 'skip' or 'mode', got {unjudged!r}")
    mode = group_mode_wrong_answer(answers, ok, uids, golds=golds)
    out, own = [], []
    for i, (a, k) in enumerate(zip(answers, ok)):
        usable = (not k) and a is not None and not (
            golds is not None and golds[i] is not None and a.strip() == str(golds[i]).strip()
        )
        if usable:
            out.append(a)
        elif (not k) and unjudged == "skip":
            out.append(None)
        else:
            out.append(mode[i])
        own.append(bool(usable))
    return out, own

def group_outcome_masks(scores: torch.Tensor, uids, threshold: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per row: does my group score identically. Returns (all_wrong, all_correct, row_ok), bool (B,)."""
    ok = scores >= threshold
    n_in: dict = {}; n_ok: dict = {}
    for i, u in enumerate(uids):
        n_in[u] = n_in.get(u, 0) + 1
        n_ok[u] = n_ok.get(u, 0) + int(bool(ok[i].item()))
    all_wrong = torch.tensor([n_ok[u] == 0 for u in uids], device=scores.device, dtype=torch.bool)
    all_correct = torch.tensor([n_ok[u] == n_in[u] for u in uids], device=scores.device, dtype=torch.bool)
    return all_wrong, all_correct, ok


def impute_group_advantage(
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    scores: torch.Tensor,
    uids,
    threshold: float,
    wrong_scale: float = 1.0,
    correct_scale: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
    """Give a group that GRPO scores identically the advantage its rollouts would have had in a
    mixed group.

    Group-relative advantages are identically zero on a group whose rollouts all scored the
    same, so an all-wrong group -- the problem the policy is stuck on -- contributes nothing to
    the update; measured, that is 34-50% of the rows. Each such rollout gets A = -wrong_scale*c,
    where c is this batch's mean |A| over the WRONG rollouts of mixed groups: the counterfactual
    "had some siblings succeeded". All-correct groups get +correct_scale*c (default 0: positive-
    only reinforcement of solved problems narrows the policy, arXiv 2506.01347). Downstream, the
    RLSD reweighting decides which tokens of an imputed rollout carry its push.

    Returns (new advantages, metrics, imputed_row_mask)."""
    all_wrong, all_correct, ok = group_outcome_masks(scores, uids, threshold)
    mask = response_mask.to(advantages.dtype)
    row_abs = (advantages.abs() * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1.0)
    wrong_in_mixed = ~all_wrong & ~all_correct & ~ok
    metrics = {                          # same group shares, same names, as all_wrong_group_rows
        "all_wrong/row_frac": all_wrong.float().mean().item(),
        "all_wrong/all_correct_row_frac": all_correct.float().mean().item(),
        "impute/c": float("nan"),
    }
    imputed = torch.zeros_like(all_wrong)
    if not wrong_in_mixed.any():
        return advantages, metrics, imputed
    c = row_abs[wrong_in_mixed].mean()
    new_adv = advantages.clone()
    if wrong_scale > 0.0 and all_wrong.any():
        new_adv[all_wrong] = (-wrong_scale * c) * mask[all_wrong]
        imputed |= all_wrong
    if correct_scale > 0.0 and all_correct.any():
        new_adv[all_correct] = (correct_scale * c) * mask[all_correct]
        imputed |= all_correct
    metrics["impute/c"] = float(c.item())
    return new_adv, metrics, imputed


def all_wrong_group_rows(
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    scores: torch.Tensor,
    uids,
    threshold: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Per row: 1.0 on the rollouts of an all-wrong group whose advantage is 0, else 0.0.

    The loss gives those rows the additive contrastive credit A_t = sipo_all_wrong_kappa *
    clip(e_t). A row of an all-wrong group whose advantage is not 0 (a shaped reward that
    differs inside the group) already goes through the multiplicative path and is not marked,
    so nothing is counted twice. All-correct groups are not marked: no wrong answer to contrast.
    Also logs c, the batch's mean |A_r| over the wrong rollouts of mixed groups, which is what
    sipo_all_wrong_kappa should be compared with: to first order those rollouts receive
    lam*|A_r|*e_t from the teacher, so kappa = lam*c is the same teacher term.

    Returns (rows (B,) float, metrics)."""
    all_wrong, all_correct, ok = group_outcome_masks(scores, uids, threshold)
    mask = response_mask.to(advantages.dtype)
    row_abs = (advantages.abs() * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1.0)
    wrong_in_mixed = ~all_wrong & ~all_correct & ~ok
    rows = all_wrong & (row_abs <= 1e-6)
    metrics = {
        "all_wrong/row_frac": all_wrong.float().mean().item(),
        "all_wrong/all_correct_row_frac": all_correct.float().mean().item(),
        "all_wrong/c": float(row_abs[wrong_in_mixed].mean().item()) if wrong_in_mixed.any() else float("nan"),
        "all_wrong/marked_row_frac": rows.float().mean().item(),
    }
    return rows.to(advantages.dtype), metrics


# Student log-prob edges for the confidence baseline: bucket 0 is lp < -4 (p < 0.02, the
# student's exploratory choices), bucket 6 is lp >= -0.05 (p > 0.95, tokens it never doubted).
_CONF_EDGES = (-4.0, -2.0, -1.0, -0.5, -0.2, -0.05)


def _confidence_baseline(
    a: torch.Tensor,
    student_lp: torch.Tensor,
    mask: torch.Tensor,
    state: Optional[dict] = None,
    ema_rate: float = 0.1,
    min_count: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Turn the raw evidence A_d into a per-token advantage by subtracting its mean over
    tokens of the same student confidence.

    E_{y~pi_S}[A_d] = -KL_t <= 0, and the KL is largest where the student is least sure, so
    the raw evidence says "the teacher is more confident" at every uncertain position --
    which under sign(A_r) becomes a push against the student's exploration whatever the
    outcome. Centring within confidence buckets removes that component; what is left is
    zero-mean at every confidence level and says only whether THIS token sat better or
    worse with the answer than the student's typical choice there. The per-bucket means
    are smoothed across micro batches with an EMA held in `state` (a value baseline).
    Returns (centred evidence, per-token baseline, metrics)."""
    m = mask.bool()
    edges = torch.tensor(_CONF_EDGES, device=a.device, dtype=torch.float32)
    nb = len(_CONF_EDGES) + 1
    bucket = torch.bucketize(student_lp.float(), edges)
    sel = bucket[m]
    sums = torch.zeros(nb, device=a.device, dtype=torch.float32).scatter_add_(0, sel, a[m].float())
    cnts = torch.zeros(nb, device=a.device, dtype=torch.float32).scatter_add_(0, sel, torch.ones_like(sel, dtype=torch.float32))
    cur = sums / cnts.clamp(min=1.0)
    have = cnts >= min_count
    if state is not None and state.get("ema") is not None:
        ema = state["ema"].to(a.device); seen = state["seen"].to(a.device)
        ema = torch.where(have, torch.where(seen, (1.0 - ema_rate) * ema + ema_rate * cur, cur), ema)
        seen = seen | have
    else:
        ema = torch.where(have, cur, torch.zeros_like(cur)); seen = have.clone()
    if state is not None:
        state["ema"] = ema.detach(); state["seen"] = seen
    base = torch.where(seen, ema, torch.zeros_like(ema))
    b_tok = base[bucket].to(a.dtype) * mask.to(a.dtype)
    centered = (a - b_tok) * mask.to(a.dtype)
    nan = float("nan")
    tot = cnts.sum().clamp(min=1.0)
    metrics = {}
    for i in range(nb):
        metrics[f"sipo/base/b{i}"] = base[i].item() if bool(seen[i]) else nan
        metrics[f"sipo/base/frac{i}"] = (cnts[i] / tot).item() if m.any() else nan
    return centered, b_tok, metrics


def _show_token(tok: str, width: int = 12) -> str:
    """Make whitespace visible and clamp the width so the table columns line up."""
    tok = tok.replace("\n", "⏎").replace("\t", "⇥").replace(" ", "␣")
    return (tok[: width - 1] + "…") if len(tok) > width else tok


def _format_sipo_main(dbg: dict, tokens: Optional[list[str]] = None, step: Optional[int] = None,
                      n_head: int = 20, n_top: int = 6, n_tail: int = 8) -> str:
    """Render one mixed-group rollout of the RLSD credit token by token so it can be checked by eye.

    Columns (both forms):
        logp_S, logp_T   student / teacher(gold) log-prob of the sampled token
        logp_T-          teacher(wrong answer) log-prob, contrastive evidence only
        A_d              the evidence: logp_T - logp_S (opd) or logp_T - logp_T- (contrastive)
        base             the evidence baseline at this token's student confidence (0 if off)
        A^               A_d - base, the evidence the credit is built from
    multiplicative form:
        w                exp(sign(A_r) * A^): > 1 where the teacher favours the token on a
                         correct rollout, or disfavours it on a wrong one
        w_clip           w clamped to [1 - eps, 1 + eps];  w_eff = (1 - lam) + lam * w_clip
        A_t              A_r * w_eff, the advantage the policy gradient sees
    additive form:
        clipA^           A^ clamped to [-eps, +eps]
        A_t              clamp_sign(A_r + kappa * clipA^): same sign as A_r, or 0
    """
    T = dbg["T"]; A_r = dbg["A_r"]; eps = dbg["eps"]; lam = dbg["lam"]
    additive = dbg.get("form") == "additive"; kap = float(dbg.get("kappa", float("nan")))
    At = dbg["A_t"]; dA = At - A_r
    kind = "CORRECT rollout (A_r > 0)" if A_r > 0 else "WRONG rollout (A_r < 0)"
    toks = [_show_token(t) for t in tokens] if tokens is not None else None
    def tokstr(i):
        if toks is not None:
            return toks[i]
        ids = dbg.get("token_ids")
        return f"id={int(ids[i])}" if ids is not None else "-"
    P = "[sipo/debug]"
    A_hat = dbg.get("A_hat", dbg["A_d"]); base = dbg.get("base", torch.zeros_like(dbg["A_d"]))
    contrastive = dbg.get("evidence") == "contrastive" and dbg.get("logp_tneg") is not None
    ref = dbg["logp_tneg"] if contrastive else dbg["logp_s"]
    ref_name = "logp_T-" if contrastive else "logp_S"
    a_clip = A_hat.clamp(-eps, eps)
    if additive:
        raw = A_r + kap * a_clip
        expect = raw.clamp(min=0.0) if A_r > 0 else raw.clamp(max=0.0)
        form_desc = f"kappa={kap:.4f} | A_t = clamp_sign(A_r + kappa*clip(A^, -eps, +eps))"
        formula_check = (f"A_t=clamp_sign(A_r+kappa*clip(A^)): max err {float((At - expect).abs().max()):.1e}   "
                         f"clamp hit {int((expect != raw).sum())}/{T} tokens (want 0 while kappa*eps < 1/G)")
        col_hdr = f" {'clipA^':>7} {'A_t':>8} {'dA':>8}"
    else:
        w_check = (dbg["w"] - torch.exp(torch.sign(torch.tensor(A_r)) * A_hat)).abs().max()
        a_check = (At - A_r * ((1 - lam) + lam * dbg["w_clip"])).abs().max()
        form_desc = f"lam={lam:.3f} | A_t = A_r*((1-lam)+lam*clip(exp(sign(A_r)*A^), 1-eps, 1+eps))"
        formula_check = (f"w=exp(sign(A_r)*A^): max err {float(w_check):.1e}   "
                         f"w_clip in [{float(dbg['w_clip'].min()):.3f}, {float(dbg['w_clip'].max()):.3f}] (want within [{1-eps:.2f}, {1+eps:.2f}])   "
                         f"A_t=A_r*((1-lam)+lam*w_clip): max err {float(a_check):.1e}")
        col_hdr = f" {'w':>7} {'w_clip':>7} {'w_eff':>7} {'A_t':>8} {'dA':>8}"
    lines = [
        f"{P} ===== step {step} | row {dbg['row']} | {kind} | A_r={A_r:+.4f} | T={T} tokens | eps={eps} {form_desc} | evidence={dbg.get('evidence', 'opd')} =====",
        f"{P} CHECKS  A_d=logp_T-{ref_name}: max err {float((dbg['A_d'] - (dbg['logp_t'] - ref)).abs().max()):.1e}   "
        f"A^=A_d-base: max err {float((A_hat - (dbg['A_d'] - base)).abs().max()):.1e}   " + formula_check,
        f"{P} {'pos':>5} {'token':<13} {'logp_S':>8} {'logp_T':>8}" + (f" {'logp_T-':>8}" if contrastive else "") +
        f" {'A_d':>8} {'base':>7} {'A^':>7}" + col_hdr,
    ]
    def row(i):
        head = (f"{P} {int(dbg['positions'][i]):>5} {tokstr(i):<13} {float(dbg['logp_s'][i]):>8.3f} {float(dbg['logp_t'][i]):>8.3f} "
                + (f"{float(dbg['logp_tneg'][i]):>8.3f} " if contrastive else "") +
                f"{float(dbg['A_d'][i]):>8.3f} {float(base[i]):>7.3f} {float(A_hat[i]):>7.3f} ")
        if additive:
            return head + f"{float(a_clip[i]):>7.3f} {float(At[i]):>8.4f} {float(dA[i]):>+8.4f}"
        return head + (f"{float(dbg['w'][i]):>7.3f} {float(dbg['w_clip'][i]):>7.3f} {float(dbg['w_eff'][i]):>7.3f} "
                       f"{float(At[i]):>8.4f} {float(dA[i]):>+8.4f}")
    for i in range(min(n_head, T)):
        lines.append(row(i))
    if T > n_head + n_tail:
        lines.append(f"{P} ... ({T - n_head - n_tail} more tokens) ... last {n_tail} (the answer):")
        for i in range(T - n_tail, T):
            lines.append(row(i))
    elif T > n_head:
        for i in range(n_head, T):
            lines.append(row(i))
    k = min(n_top, T)
    up = torch.topk(dA, k=k).indices.tolist(); down = torch.topk(-dA, k=k).indices.tolist()
    def brief(i):
        return f"{tokstr(i)}@{int(dbg['positions'][i])}({float(dA[i]):+.4f}, A^={float(A_hat[i]):+.2f})"
    lines.append(f"{P} CREDIT UP   (teacher favours; pushed {'up harder' if A_r > 0 else 'down less'}): " + "  ".join(brief(i) for i in up))
    lines.append(f"{P} CREDIT DOWN (teacher disfavours; pushed {'up less' if A_r > 0 else 'down harder'}): " + "  ".join(brief(i) for i in down))
    nz = int((dbg["A_d"].abs() > 1e-3).sum())
    at_clip = int((A_hat.abs() >= eps).sum()) if additive else int(((dbg["w"] < 1 - eps) | (dbg["w"] > 1 + eps)).sum())
    lines.append(f"{P} tokens with |A_d|>1e-3: {nz}/{T} ({100.0*nz/T:.0f}%)   at clip: {at_clip}/{T}   "
                 f"sum|dA| = {float(dA.abs().sum()):.4f} of sum|A_r| = {T*abs(A_r):.4f} ({100*float(dA.abs().sum())/(T*abs(A_r)):.1f}%)")
    return "\n".join(lines)

def _format_sipo_all_wrong(dbg: dict, tokens: Optional[list[str]] = None, step: Optional[int] = None,
                           n_head: int = 12, n_top: int = 6, n_tail: int = 8) -> str:
    """Render one rollout of an all-wrong group under sipo_all_wrong_kappa: A_t = kappa * clip(e_t)."""
    T = dbg["aw_T"]; kap = dbg["aw_kappa"]; eps = dbg["aw_eps"]
    e = dbg["aw_e"]; A = dbg["aw_A"]
    toks = [_show_token(t) for t in tokens] if tokens is not None else None
    def tokstr(i):
        if toks is not None:
            return toks[i]
        ids = dbg.get("aw_token_ids")
        return f"id={int(ids[i])}" if ids is not None else "-"
    P = "[sipo/debug]"
    a_check = (A - kap * e.clamp(-eps, eps)).abs().max()
    lines = [
        f"{P} ===== step {step} | row {dbg['aw_row']} | ALL-WRONG group (A_r = 0) | kappa={kap:.4f} "
        f"| T={T} tokens | eps={eps} | A_t = kappa*clip(e_t, -eps, +eps) =====",
        f"{P} CHECKS  A_t=kappa*clip(e): max err {float(a_check):.1e}   |A_t| <= kappa*eps = {kap*eps:.4f}: max|A_t| = {float(A.abs().max()):.4f}",
        f"{P} {'pos':>5} {'token':<13} {'logp_T':>8} {'logp_T-':>8} {'e':>8} {'A_t':>8}",
    ]
    def row(i):
        return (f"{P} {int(dbg['aw_positions'][i]):>5} {tokstr(i):<13} {float(dbg['aw_logp_t'][i]):>8.3f} "
                f"{float(dbg['aw_logp_tneg'][i]):>8.3f} {float(e[i]):>8.3f} {float(A[i]):>+8.4f}")
    for i in range(min(n_head, T)):
        lines.append(row(i))
    if T > n_head + n_tail:
        lines.append(f"{P} ... ({T - n_head - n_tail} more tokens) ... last {n_tail} (the answer):")
        for i in range(T - n_tail, T):
            lines.append(row(i))
    elif T > n_head:
        for i in range(n_head, T):
            lines.append(row(i))
    k = min(n_top, T)
    up = torch.topk(A, k=k).indices.tolist(); down = torch.topk(-A, k=k).indices.tolist()
    def brief(i):
        return f"{tokstr(i)}@{int(dbg['aw_positions'][i])}({float(A[i]):+.4f}, e={float(e[i]):+.2f})"
    lines.append(f"{P} CREDIT UP   (gold-like, pushed up):    " + "  ".join(brief(i) for i in up))
    lines.append(f"{P} CREDIT DOWN (wrong-like, pushed down): " + "  ".join(brief(i) for i in down))
    nz = int((e.abs() > 1e-3).sum()); at_clip = int((e.abs() >= eps).sum())
    lines.append(f"{P} tokens with |e|>1e-3: {nz}/{T} ({100.0*nz/T:.0f}%)   at clip: {at_clip}/{T}   "
                 f"sum A_t = {float(A.sum()):+.4f} (sequence-level push; < 0: this rollout reads as the wrong answer)   "
                 f"sum|A_t| = {float(A.abs().sum()):.4f}")
    return "\n".join(lines)


def format_sipo_debug(dbg: dict, tokens: Optional[list[str]] = None, step: Optional[int] = None,
                      n_head: int = 20, n_top: int = 6, n_tail: int = 8,
                      aw_tokens: Optional[list[str]] = None) -> str:
    """Render the RLSD debug row(s): the reweighted row of a mixed group (keys row/...), and,
    under sipo_all_wrong_kappa, one row of an all-wrong group (keys aw_row/...)."""
    parts = []
    if "row" in dbg:
        parts.append(_format_sipo_main(dbg, tokens=tokens, step=step, n_head=n_head, n_top=n_top, n_tail=n_tail))
    if "aw_row" in dbg:
        parts.append(_format_sipo_all_wrong(dbg, tokens=aw_tokens, step=step, n_top=n_top, n_tail=n_tail))
    return "\n".join(parts)


def compute_self_instruction_loss(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    self_distillation_config: Any,
    advantages: torch.Tensor,
    old_log_probs: Optional[torch.Tensor] = None,
    student_all_log_probs: Optional[torch.Tensor] = None,
    teacher_all_log_probs: Optional[torch.Tensor] = None,
    student_topk_log_probs: Optional[torch.Tensor] = None,
    teacher_topk_log_probs: Optional[torch.Tensor] = None,
    self_instruction_mask: Optional[torch.Tensor] = None,
    teacher_neg_log_probs: Optional[torch.Tensor] = None,
    loss_agg_mode: str = "token-mean",
    rollout_is_weights: Optional[torch.Tensor] = None,
    clip_ratio: float = 0.2,
    clip_ratio_low: Optional[float] = None,
    clip_ratio_high: Optional[float] = None,
    clip_ratio_c: float = 3.0,
    global_step: Optional[int] = None,
    debug: Optional[dict] = None,
    baseline_state: Optional[dict] = None,
    all_wrong_row: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Self-instruction objective with PPO-style reward and beta-scaled divergence.

    loss = reward_loss + beta * divergence_loss

    With `self_distillation.reinforce_advantage=True` (requires alpha=1.0 and
    full_logit_distillation=False) the divergence term is dropped and the teacher instead
    reweights the reward advantage within each rollout -- RLSD, arXiv:2604.03128:

        A_d  = log pi_teacher(y_t | x, r) - log pi_student(y_t | x)        (detached)
        w_t  = clip(exp(sign(A_r) * A_d), 1 - eps, 1 + eps)
        A_t  = A_r * ((1 - lam) + lam * w_t),   lam: sipo_lambda_start -> 0 over sipo_lambda_steps

    The verifier alone decides whether a rollout is pushed up or down; the teacher, which
    has seen the privileged context r, only moves credit between its tokens. `beta` plays
    no part in this arm.
    """

    reward_mask = response_mask
    divergence_mask = response_mask
    if self_instruction_mask is not None:
        divergence_mask = divergence_mask * self_instruction_mask.unsqueeze(1)
    beta = float(getattr(self_distillation_config, "beta", 1.0))
    if beta < 0.0:
        raise ValueError(f"self_distillation.beta must be non-negative, got {beta}")

    # A_d = log pi_teacher(y_t | x, r) - log pi_student(y_t | x): the teacher's evidence for
    # the sampled token once it has seen the privileged context r. Always logged for
    # loss_mode="sipo"; it only touches the advantage when reinforce_advantage is on.
    use_reinforce_adv = bool(getattr(self_distillation_config, "reinforce_advantage", False))
    distill_adv = (teacher_log_probs - student_log_probs).detach()
    evidence_metrics = _evidence_stats(distill_adv, divergence_mask, "opd")
    with torch.no_grad():
        reward_abs = advantages[reward_mask.bool()].float().abs()      # |A_r| before the teacher term
        reward_adv_abs_mean = reward_abs.mean().item() if reward_abs.numel() else float("nan")

    if use_reinforce_adv:
        if self_distillation_config.alpha != 1.0:
            raise ValueError(
                "reinforce_advantage reweights the reward advantage with the reverse-KL log-ratio "
                f"and only exists for self_distillation.alpha=1.0, got {self_distillation_config.alpha}."
            )
        if self_distillation_config.full_logit_distillation:
            raise ValueError(
                "reinforce_advantage replaces the analytic divergence term; set "
                "self_distillation.full_logit_distillation=False. That also skips the top-k "
                "logit tensors, which is what makes model.use_fused_kernels=True usable."
            )
        # RLSD (arXiv:2604.03128, Algorithm 1). The verifier keeps sole authority over the
        # DIRECTION of every token's update; the teacher only redistributes MAGNITUDE within
        # a rollout, by at most +-eps_w:
        #     w_t = exp(sign(A_r) * A_d)                     evidence ratio, inverted on a wrong rollout
        #     A_t = A_r * ((1 - lam) + lam * clip(w_t, 1 - eps_w, 1 + eps_w))
        # lam starts at sipo_lambda_start and decays linearly to 0 over sipo_lambda_steps, so
        # the reweighting fades out and the update returns to plain GRPO (the paper's 0.5 -> 0
        # over 50 steps). Rows with A_r = 0 stay at 0 whatever the teacher says.
        eps_w = float(getattr(self_distillation_config, "sipo_eps", 0.2))
        lam0 = float(getattr(self_distillation_config, "sipo_lambda_start", 0.5))
        lam_steps = int(getattr(self_distillation_config, "sipo_lambda_steps", 50))
        # Same clock as the reference implementation (iie-ycx/RLSD, opsd_trainer.py): the
        # trainer counts the first update as step 1, and lam = lam0 * (1 - step / steps).
        lam = lam0
        if global_step is not None and lam_steps > 0:
            lam = lam0 * max(0.0, 1.0 - float(global_step) / float(lam_steps))
        m = divergence_mask.bool()
        sgn = torch.sign(advantages)
        # sipo_evidence: what the weight is built from.
        #   opd          e_t = log pi_T(y_t | x, gold) - log pi_S(y_t | x)      (RLSD as published)
        #   contrastive  e_t = log pi_T(y_t | x, gold) - log pi_T(y_t | x, wrong answer)
        # The opd evidence is the per-token integrand of KL(pi_S || pi_T): its expectation is
        # -KL_t and the reweighting it induces is lam*|A_r| times the on-policy-distillation
        # gradient toward the answer-conditioned model, i.e. "act as if you knew the answer" --
        # decisive, terse, less exploration -- which on long-CoT math cancels GRPO's own
        # progress. In the contrastive evidence BOTH sides are answer-conditioned, so that
        # decisiveness prior cancels; e_t is ~0 on every token that does not depend on which
        # answer is in the context (deliberation, checks, format) and nonzero only where the
        # token is specific to the right answer (e_t > 0) or to the wrong one (e_t < 0).
        evidence_mode = str(getattr(self_distillation_config, "sipo_evidence", "opd"))
        contrast_metrics = {}
        if evidence_mode == "contrastive":
            if teacher_neg_log_probs is None:
                raise ValueError("sipo_evidence=contrastive needs teacher_neg_log_probs (the wrong-answer teacher).")
            raw_evidence = (teacher_log_probs - teacher_neg_log_probs).detach()
            contrast_metrics = {k: v for k, v in _evidence_stats(raw_evidence, divergence_mask, "contrast").items()
                                if k.split("/")[1] in ("rms", "p01", "p50", "p99", "head_tail_gap", "mean")}
            with torch.no_grad():
                sel = raw_evidence[divergence_mask.bool()]
                contrast_metrics["contrast/nonzero_frac"] = (sel.abs() > 1e-3).float().mean().item() if sel.numel() else float("nan")
        else:
            raw_evidence = distill_adv
        # sipo_baseline=confidence: subtract the evidence's mean at the student's confidence
        # level (see _confidence_baseline). Composable with either evidence.
        baseline_mode = str(getattr(self_distillation_config, "sipo_baseline", "none"))
        evidence, base_tok, base_metrics = raw_evidence, torch.zeros_like(raw_evidence), {}
        if baseline_mode == "confidence":
            evidence, base_tok, base_metrics = _confidence_baseline(
                raw_evidence, student_log_probs.detach(), divergence_mask, state=baseline_state,
                ema_rate=float(getattr(self_distillation_config, "sipo_baseline_ema", 0.1)),
            )
        # sipo_negative_only (reference repo flag): reweight wrong rollouts only; correct
        # rollouts keep GRPO's uniform credit. In the additive form "wrong" includes the rows
        # of an all-wrong group (A_r = 0).
        neg_only = bool(getattr(self_distillation_config, "sipo_negative_only", False))
        # sipo_form: how the evidence enters the advantage.
        #   multiplicative  A_t = A_r * ((1-lam) + lam * clip(exp(sign(A_r) e_t), 1-eps, 1+eps))
        #                   (RLSD / CEPO). First order A_r + lam*|A_r|*e_t: the teacher term
        #                   scales with |A_r| and is 0 on all-wrong groups unless
        #                   sipo_all_wrong_kappa adds it back there with a constant (below).
        #   additive        A_t = A_r + kappa * clip(e_t, -eps, +eps) for EVERY row (RLCSD's
        #                   form, arXiv:2606.11709): the same constant kappa on a wrong rollout
        #                   of a 1-of-8 group (multiplicative: lam/8) and on an all-wrong group
        #                   (multiplicative: 0). A sign-preserving clamp (RLCSD eq. 11) keeps
        #                   sign(A_t) = sign(A_r) on mixed rows; it cannot engage while
        #                   kappa*eps < min |A_r| = 1/G, and sipo/clamp_frac reports if it does.
        #                   Rows with A_r = 0 keep the sign of e_t: nothing to anchor to, and
        #                   that is the all-wrong signal. lam is unused.
        form = str(getattr(self_distillation_config, "sipo_form", "multiplicative"))
        e_clip = evidence.clamp(-eps_w, eps_w)
        nan = float("nan")
        kappa = nan
        if form == "additive":
            if evidence_mode != "contrastive":
                raise ValueError("sipo_form=additive needs sipo_evidence=contrastive: the opd evidence has mean -KL_t on every token.")
            kappa = float(getattr(self_distillation_config, "sipo_kappa", 0.0))
            if kappa <= 0.0:
                raise ValueError("sipo_form=additive needs sipo_kappa > 0.")
            apply = m & (advantages <= 0) if neg_only else m
            raw_sum = advantages + kappa * e_clip * apply.to(evidence.dtype)
            new_adv = torch.where(sgn > 0, raw_sum.clamp(min=0.0), torch.where(sgn < 0, raw_sum.clamp(max=0.0), raw_sum))
            clamp_hit = apply & (new_adv != raw_sum)
            w_raw = w_clip = w_eff = None
        elif form == "multiplicative":
            apply = m & (advantages < 0) if neg_only else m
            w_raw = torch.exp((sgn * evidence).clamp(-20.0, 20.0))
            w_clip = w_raw.clamp(1.0 - eps_w, 1.0 + eps_w)
            w_eff = torch.where(apply, (1.0 - lam) + lam * w_clip, torch.ones_like(w_clip))
            new_adv = advantages * w_eff
            clamp_hit = torch.zeros_like(apply)
        else:
            raise ValueError(f"sipo_form must be 'multiplicative' or 'additive', got {form!r}")
        active = apply & (sgn != 0)                       # tokens of mixed-group rows
        # Idle micro batches (no reprompted token on a row with A_r != 0) report NaN, which
        # reduce_metrics ignores; sipo/active stays 0/1 so the active fraction is visible.
        sipo_metrics = {
            "sipo/form_additive": 1.0 if form == "additive" else 0.0,
            "sipo/lambda": nan if form == "additive" else lam, "sipo/kappa": kappa, "sipo/active": 0.0,
            "sipo/w_mean": nan, "sipo/w_at_low_frac": nan, "sipo/w_at_high_frac": nan,
            "sipo/w_head_tail_gap": nan, "sipo/w_eff_rms_dev": nan, "sipo/dA_rms": nan,
            "sipo/evidence_rms": nan, "sipo/at_clip_frac": nan, "sipo/clamp_frac": nan,
            **base_metrics,
            **contrast_metrics,
        }
        if active.any():
            dA = new_adv - advantages
            sipo_metrics.update({
                "sipo/active": 1.0,
                "sipo/dA_rms": dA[active].pow(2).mean().sqrt().item(),
                "sipo/evidence_rms": evidence[active].float().pow(2).mean().sqrt().item(),
                "sipo/at_clip_frac": (evidence[active].abs() >= eps_w).float().mean().item(),
                "sipo/clamp_frac": clamp_hit[active].float().mean().item(),
            })
            if w_clip is not None:
                mf = m.float()
                frac_pos = (mf.cumsum(dim=-1) - 1.0) / mf.sum(dim=-1, keepdim=True).clamp(min=1.0)
                head = w_clip[active & (frac_pos < 0.25)]; tail = w_clip[active & (frac_pos >= 0.75)]
                sipo_metrics.update({
                    "sipo/w_mean": w_clip[active].mean().item(),
                    "sipo/w_at_low_frac": (w_raw[active] <= 1.0 - eps_w).float().mean().item(),
                    "sipo/w_at_high_frac": (w_raw[active] >= 1.0 + eps_w).float().mean().item(),
                    "sipo/w_head_tail_gap": (tail.mean() - head.mean()).item() if head.numel() and tail.numel() else nan,
                    "sipo/w_eff_rms_dev": (w_eff[active] - 1.0).pow(2).mean().sqrt().item(),
                })
            if debug is not None and not debug:
                r = int(torch.nonzero(active.any(dim=-1), as_tuple=False).flatten()[0].item())
                pos = m[r]
                ones = torch.ones(int(pos.sum().item()))
                debug.update({
                    "row": r, "A_r": float(advantages[r][pos][0].item()), "T": int(pos.sum().item()),
                    "eps": eps_w, "lam": lam, "form": form, "kappa": kappa,
                    "positions": torch.nonzero(pos).flatten().cpu(),
                    "logp_s": student_log_probs.detach()[r][pos].float().cpu(),
                    "logp_t": teacher_log_probs.detach()[r][pos].float().cpu(),
                    "A_d": raw_evidence[r][pos].float().cpu(), "base": base_tok[r][pos].float().cpu(),
                    "evidence": evidence_mode,
                    "logp_tneg": teacher_neg_log_probs.detach()[r][pos].float().cpu() if teacher_neg_log_probs is not None else None,
                    "A_hat": evidence[r][pos].float().cpu(),
                    "w": w_raw[r][pos].float().cpu() if w_raw is not None else ones,
                    "w_clip": w_clip[r][pos].float().cpu() if w_clip is not None else ones,
                    "w_eff": w_eff[r][pos].float().cpu() if w_eff is not None else ones,
                    "A_t": new_adv[r][pos].float().cpu(),
                })
        # ---- rows of all-wrong groups (A_r = 0) ----
        # additive form: they went through the one formula above (A_r = 0 -> kappa*clip(e_t)).
        # multiplicative form: sipo_all_wrong_kappa > 0 adds A_t = sipo_all_wrong_kappa *
        # clip(e_t) on the rows the trainer marked (all_wrong_row): to first order the teacher
        # term lam*|A_r|*e_t a wrong rollout in a mixed group receives (|A_r| ~ c, logged as
        # all_wrong/c), minus the constant -c that imputation added and that was measured to be
        # pure variance. Contrastive evidence only. Either way these rows get their own
        # metrics (all_wrong/*) and one debug row.
        aw_kappa = float(getattr(self_distillation_config, "sipo_all_wrong_kappa", 0.0))
        aw_apply = torch.zeros_like(m)
        aw_adv = torch.zeros_like(new_adv)
        aw_k = nan
        if form == "additive":
            if all_wrong_row is not None:
                zero_rows = all_wrong_row > 0
            else:
                zero_rows = m.any(dim=-1) & ~((advantages != 0) & m).any(dim=-1)
            aw_apply = zero_rows[:, None] & apply
            aw_adv = torch.where(aw_apply, new_adv - advantages, torch.zeros_like(new_adv))
            aw_k = kappa
        elif aw_kappa > 0.0:
            if evidence_mode != "contrastive":
                raise ValueError("sipo_all_wrong_kappa > 0 needs sipo_evidence=contrastive.")
            if all_wrong_row is not None:
                aw_apply = (all_wrong_row > 0)[:, None] & m
                aw_adv = aw_kappa * e_clip * aw_apply.to(evidence.dtype)
                new_adv = new_adv + aw_adv
            aw_k = aw_kappa
        # dA = the teacher term on these rows (their A_r is 0, so dA is also their A_t),
        # the counterpart of sipo/dA_rms on mixed rows.
        aw_metrics = {
            "all_wrong/active": 0.0, "all_wrong/kappa": nan, "all_wrong/dA_rms": nan, "all_wrong/dA_mean": nan,
            "all_wrong/row_sum_mean": nan, "all_wrong/nonzero_frac": nan, "all_wrong/at_clip_frac": nan,
        }
        if aw_apply.any():
            sel = aw_adv[aw_apply]
            rows = aw_apply.any(dim=-1)
            aw_metrics.update({
                "all_wrong/active": 1.0,
                "all_wrong/kappa": aw_k,
                "all_wrong/dA_rms": sel.pow(2).mean().sqrt().item(),
                "all_wrong/dA_mean": sel.mean().item(),
                "all_wrong/row_sum_mean": aw_adv.sum(dim=-1)[rows].mean().item(),
                "all_wrong/nonzero_frac": (evidence[aw_apply].abs() > 1e-3).float().mean().item(),
                "all_wrong/at_clip_frac": (evidence[aw_apply].abs() >= eps_w).float().mean().item(),
            })
            if debug is not None and "aw_row" not in debug:
                r = int(torch.nonzero(rows, as_tuple=False).flatten()[0].item())
                pos = aw_apply[r]
                debug.update({
                    "aw_row": r, "aw_kappa": aw_k, "aw_eps": eps_w,
                    "aw_T": int(pos.sum().item()),
                    "aw_positions": torch.nonzero(pos).flatten().cpu(),
                    "aw_logp_t": teacher_log_probs.detach()[r][pos].float().cpu(),
                    "aw_logp_tneg": teacher_neg_log_probs.detach()[r][pos].float().cpu(),
                    "aw_e": evidence[r][pos].float().cpu(),
                    "aw_A": aw_adv[r][pos].float().cpu(),
                })
        sipo_metrics.update(aw_metrics)
        evidence_metrics.update(sipo_metrics)
        advantages = new_adv

    # 1) Reward term: use externally-computed token-level advantages
    # (e.g., from GRPO) with PPO-style clipping.
    if clip_ratio_low is None:
        clip_ratio_low = clip_ratio
    if clip_ratio_high is None:
        clip_ratio_high = clip_ratio
    if clip_ratio_c <= 1.0:
        raise ValueError(f"clip_ratio_c must be > 1.0, got {clip_ratio_c}.")

    old_for_pg = old_log_probs if old_log_probs is not None else student_log_probs.detach()
    negative_approx_kl = student_log_probs - old_for_pg
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)

    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)

    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    reward_loss = agg_loss(
        loss_mat=pg_losses,
        loss_mask=reward_mask,
        loss_agg_mode=loss_agg_mode,
        batch_num_tokens=reward_mask.sum().clamp(min=1.0),
    )

    # 2) Divergence term: same alpha family as compute_self_distillation_loss.
    if self_distillation_config.full_logit_distillation:
        use_topk = self_distillation_config.distillation_topk is not None
        if use_topk:
            if student_topk_log_probs is None or teacher_topk_log_probs is None:
                raise ValueError(
                    "self-instruction top-k distillation requires student_topk_log_probs and teacher_topk_log_probs."
                )

            def add_tail(log_probs: torch.Tensor) -> torch.Tensor:
                log_s = torch.logsumexp(log_probs, dim=-1, keepdim=True)
                log_s = torch.clamp(log_s, max=-1e-7)
                tail_log = torch.log(-torch.expm1(log_s))
                return torch.cat([log_probs, tail_log], dim=-1)

            def renorm_topk_log_probs(logp: torch.Tensor) -> torch.Tensor:
                logZ = torch.logsumexp(logp, dim=-1, keepdim=True)
                return logp - logZ

            student_distill_log_probs = student_topk_log_probs
            teacher_distill_log_probs = teacher_topk_log_probs.detach()
            if self_distillation_config.distillation_add_tail:
                student_distill_log_probs = add_tail(student_distill_log_probs)
                teacher_distill_log_probs = add_tail(teacher_distill_log_probs)
            else:
                student_distill_log_probs = renorm_topk_log_probs(student_distill_log_probs)
                teacher_distill_log_probs = renorm_topk_log_probs(teacher_distill_log_probs)
        else:
            if student_all_log_probs is None or teacher_all_log_probs is None:
                raise ValueError(
                    "self-instruction full-logit distillation requires student_all_log_probs and teacher_all_log_probs."
                )
            student_distill_log_probs = student_all_log_probs
            teacher_distill_log_probs = teacher_all_log_probs.detach()

        if self_distillation_config.alpha == 0.0:
            divergence_token = F.kl_div(
                student_distill_log_probs, teacher_distill_log_probs, reduction="none", log_target=True
            ).sum(-1)
        elif self_distillation_config.alpha == 1.0:
            divergence_token = F.kl_div(
                teacher_distill_log_probs, student_distill_log_probs, reduction="none", log_target=True
            ).sum(-1)
        else:
            alpha_t = torch.tensor(
                self_distillation_config.alpha,
                dtype=student_distill_log_probs.dtype,
                device=student_distill_log_probs.device,
            )
            mixture_log_probs = torch.logsumexp(
                torch.stack(
                    [
                        student_distill_log_probs + torch.log(1 - alpha_t),
                        teacher_distill_log_probs + torch.log(alpha_t),
                    ]
                ),
                dim=0,
            )
            kl_teacher = F.kl_div(mixture_log_probs, teacher_distill_log_probs, reduction="none", log_target=True)
            kl_student = F.kl_div(mixture_log_probs, student_distill_log_probs, reduction="none", log_target=True)
            divergence_token = torch.lerp(kl_student, kl_teacher, alpha_t).sum(-1)
    elif use_reinforce_adv:
        # Gradient is already carried by `advantages`; no separate divergence term.
        divergence_token = torch.zeros_like(student_log_probs)
    else:
        # Keep fallback simple and aligned with existing distillation behavior.
        assert self_distillation_config.alpha == 1.0, "Only reverse KL is supported for non-full-logit distillation"
        log_ratio = student_log_probs - teacher_log_probs.detach()
        divergence_token = log_ratio.detach() * student_log_probs

    # Match compute_self_distillation_loss behavior: apply IS and rollout weights
    # to distillation/divergence term as well.
    is_clip = self_distillation_config.is_clip
    if is_clip is not None:
        if old_log_probs is None:
            raise ValueError("old_log_probs is required for distillation IS ratio.")
        negative_approx_kl = (student_log_probs - old_log_probs).detach()
        negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
        ratio = torch.exp(negative_approx_kl).clamp(max=is_clip)
        divergence_token = divergence_token * ratio
    if rollout_is_weights is not None:
        divergence_token = divergence_token * rollout_is_weights

    if divergence_mask.sum() > 0:
        divergence_loss = agg_loss(
            loss_mat=divergence_token,
            loss_mask=divergence_mask,
            loss_agg_mode=loss_agg_mode,
            batch_num_tokens=divergence_mask.sum().clamp(min=1.0),
        )
    else:
        divergence_loss = divergence_token.sum() * 0.0
    scaled_divergence_loss = beta * divergence_loss
    loss = reward_loss + scaled_divergence_loss

    eps = 1e-8
    # adv_mean / pos_adv_mean / neg_adv_mean: the advantage the policy gradient used (after the
    # teacher term); reward_adv_abs_mean: |A_r| before it.
    metrics = {
        "sipo/reward_loss": reward_loss.item(),
        "sipo/divergence_loss": divergence_loss.item(),
        "sipo/scaled_divergence_loss": scaled_divergence_loss.item(),
        "sipo/divergence_beta": beta,
        "sipo/total_loss": loss.item(),
        "sipo/adv_mean": (advantages * reward_mask).sum().item() / (reward_mask.sum().item() + eps),
        "sipo/pos_adv_mean": (torch.clamp(advantages, min=0.0) * reward_mask).sum().item()
        / (reward_mask.sum().item() + eps),
        "sipo/neg_adv_mean": (torch.clamp(-advantages, min=0.0) * reward_mask).sum().item()
        / (reward_mask.sum().item() + eps),
        "sipo/reward_adv_abs_mean": reward_adv_abs_mean,
        "sipo/teacher_token_frac": divergence_mask.sum().item() / (reward_mask.sum().item() + eps),
        "sipo/reinforce_advantage": float(use_reinforce_adv),
    }
    metrics.update(evidence_metrics)
    return loss, metrics


@deprecated("verl.trainer.ppo.core_algos.compute_policy_loss_vanilla")
def compute_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
):
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        clip_ratio_c (float, optional):
            Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
            Defaults to 3.0.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
    """
    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - cliprange_low, 1 + cliprange_high
    )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(
        pg_losses1, pg_losses2
    )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


@register_policy_loss("vanilla")  # type: ignore[arg-type]
def compute_policy_loss_vanilla(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        config: `(verl.trainer.config.ActorConfig)`:
            config for the actor.
        rollout_log_probs: `(torch.Tensor)`:
            log probabilities of actions under the rollout policy, shape (batch_size, response_length).
    """

    assert config is not None
    assert not isinstance(config, AlgoConfig)
    clip_ratio = config.clip_ratio  # Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
    clip_ratio_c = config.get(  # Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
        "clip_ratio_c", 3.0
    )

    cliprange = clip_ratio
    cliprange_low = clip_ratio_low
    cliprange_high = clip_ratio_high

    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - cliprange_low, 1 + cliprange_high
    )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(
        pg_losses1, pg_losses2
    )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )

    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("gspo")
def compute_policy_loss_gspo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for GSPO.

    See https://arxiv.org/pdf/2507.18071 for more details.

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. For GSPO, it is recommended to use "seq-mean-token-mean".
    """

    assert config is not None
    assert isinstance(config, ActorConfig)
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else config.clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else config.clip_ratio

    negative_approx_kl = log_prob - old_log_prob

    # compute sequence-level importance ratio:
    # si(θ) = (π_θ(yi|x)/π_θold(yi|x))^(1/|yi|) =
    # exp [(1/|y_i|) * Σ_t log(π_θ(y_i,t|x,y_i,<t)/π_θold(y_i,t|x,y_i,<t))]
    seq_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)
    negative_approx_kl_seq = torch.sum(negative_approx_kl * response_mask, dim=-1) / seq_lengths

    # Combined ratio at token level:
    # s_i,t(θ) = sg[s_i(θ)] · π_θ(y_i,t|x, y_i,<t) / sg[π_θ(y_i,t|x, y_i,<t)]
    # In log space: log(s_i,t(θ)) = sg[log(s_i(θ))] + log_prob - sg[log_prob]
    log_seq_importance_ratio = log_prob - log_prob.detach() + negative_approx_kl_seq.detach().unsqueeze(-1)
    log_seq_importance_ratio = torch.clamp(log_seq_importance_ratio, max=10.0)  # clamp for numerical stability

    # finaly exp() to remove log
    seq_importance_ratio = torch.exp(log_seq_importance_ratio)

    pg_losses1 = -advantages * seq_importance_ratio
    pg_losses2 = -advantages * torch.clamp(seq_importance_ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    # for GSPO, we need to aggregate the loss at the sequence level (seq-mean-token-mean)
    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode="seq-mean-token-mean", **config.global_batch_info
    )

    # For compatibility, return zero for pg_clipfrac_lower (not used in standard GSPO)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)

    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("sapo")
def compute_policy_loss_sapo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the smoothed policy objective and related metrics for SAPO.

    See https://arxiv.org/pdf/2511.20347 for more details.

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. For SAPO, it is recommended to use "seq-mean-token-mean".
    """

    assert config is not None
    assert isinstance(config, ActorConfig)

    # temperature for positive and negative token updates
    tau_pos = torch.as_tensor(config.tau_pos, dtype=advantages.dtype, device=advantages.device)
    tau_neg = torch.as_tensor(config.tau_neg, dtype=advantages.dtype, device=advantages.device)

    def gate_function(x, tau):
        """The gating function used in SAPO"""
        return torch.sigmoid(tau * (x - 1.0)) * (4.0 / tau)

    # compute IS at token level:
    # r_{i,t}(θ) = π_θ(y_{i,t}|x, y_{i,<t}) / π_θold(y_{i,t}|x, y_{i,<t})]
    # In log space: log(r_{i,t}(θ)) = log_prob - ol_log_prob
    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    # finally exp() to remove log and get r_{i,t}(θ)
    ratio = torch.exp(negative_approx_kl)

    # tau_{i,t} is tau_pos if adv > 0 else tau_neg
    taus = torch.where(
        condition=advantages > 0,
        input=tau_pos,  # if A_{i,t} > 0 we set to tau_pos
        other=tau_neg,  # if A_{i,t} <= 0 we set to tau_neg
    )

    # compute the gates f_{i,t}(r_{i,t}(θ)) at token level
    gates = gate_function(ratio, taus)

    # compute policy gradient loss
    pg_losses = -gates * advantages

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    # for SAPO, we need to aggregate the loss at the sequence level (seq-mean-token-mean)
    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode="seq-mean-token-mean", **config.global_batch_info
    )

    # For compatibility, return zero for both pg_clipfrac and pg_clipfrac_lower (not used in SAPO)
    pg_clipfrac = torch.tensor(0.0, device=pg_loss.device)
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)
    # compute KL for metrics tracking
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    # return metrics dict
    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }

    return pg_loss, pg_metrics


@register_policy_loss("gpg")
def compute_policy_loss_gpg(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Adapted from
    https://github.com/AMAP-ML/GPG/blob/main/VisualThinker-R1-Zero/src/open-r1-multimodal/src/open_r1/trainer/grpo_trainer.py#L495
    Args:
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    return:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via GPG
    """
    assert config is not None
    pg_losses = -log_prob * advantages

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )
    return pg_loss, {}


@register_policy_loss("clip_cov")
def compute_policy_loss_clip_cov(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for Clip-Cov.

    Adapted from
    https://github.com/PRIME-RL/Entropy-Mechanism-of-RL/blob/main/verl/trainer/ppo/core_algos.py

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        clip_cvo_ratio (float, optional):
            Ratio for clipping the covariance. Defaults to 0.0002.
        clip_cov_lb (float, optional):
            Lower bound for clipping covariance. Defaults to 1.0.
        clip_cov_ub (float, optional):
            Upper bound for clipping covariance. Defaults to 5.0.
    """
    assert config is not None
    assert not isinstance(config, AlgoConfig), "passing AlgoConfig not supported yet"
    assert config.policy_loss is not None

    clip_cov_ratio = config.policy_loss.clip_cov_ratio if config.policy_loss.clip_cov_ratio is not None else 0.0002
    cliprange = config.clip_ratio
    cliprange_low = config.clip_ratio_low if config.clip_ratio_low is not None else cliprange
    cliprange_high = config.clip_ratio_high if config.clip_ratio_high is not None else cliprange
    clip_cov_ub = config.policy_loss.clip_cov_ub if config.policy_loss.clip_cov_ub is not None else 5.0
    clip_cov_lb = config.policy_loss.clip_cov_lb if config.policy_loss.clip_cov_lb is not None else 1.0

    assert clip_cov_ratio > 0, "clip_ratio should be larger than 0."

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio

    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange

    corr = torch.ones_like(advantages)
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
    clip_by_origin = (pg_losses2 > pg_losses1) & (response_mask > 0)

    cov_all = (advantages - verl_F.masked_mean(advantages, response_mask)) * (
        log_prob - verl_F.masked_mean(log_prob.detach(), response_mask)
    )
    cov_all[response_mask == 0] = -torch.inf
    cov_all[clip_by_origin] = -torch.inf

    clip_num = max(int(clip_cov_ratio * response_mask.sum().item()), 1)
    top_k_idx = (cov_all < clip_cov_ub) & (cov_all > clip_cov_lb) & (response_mask > 0)
    top_k_idx = torch.nonzero(top_k_idx)

    if len(top_k_idx) > 0:
        perm = torch.randperm(len(top_k_idx))
        top_k_idx = top_k_idx[perm[: min(clip_num, len(top_k_idx))]]
    else:
        top_k_idx = torch.empty((0, 2), device=cov_all.device, dtype=torch.long)

    corr[top_k_idx[:, 0], top_k_idx[:, 1]] = 0

    pg_clipfrac = verl_F.masked_mean((corr == 0).float(), response_mask)

    pg_losses = torch.maximum(pg_losses1, pg_losses2) * corr

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )
    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("kl_cov")
def compute_policy_loss_kl_cov(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for Clip-Cov.

    Adapted from
    https://github.com/PRIME-RL/Entropy-Mechanism-of-RL/blob/main/verl/trainer/ppo/core_algos.py

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        kl_cov_ratio (float, optional):
            Ratio for selecting the top-k covariance values. Defaults to 0.0002.
        ppo_kl_coef (float, optional):
            Coefficient for the KL penalty term in the loss. Defaults to 1.
    """
    assert config is not None
    assert not isinstance(config, AlgoConfig), "passing AlgoConfig not supported yet"
    assert config.policy_loss is not None

    kl_cov_ratio = config.policy_loss.kl_cov_ratio if config.policy_loss.kl_cov_ratio is not None else 0.0002
    ppo_kl_coef = config.policy_loss.ppo_kl_coef if config.policy_loss.ppo_kl_coef is not None else 1.0

    assert kl_cov_ratio > 0, "kl_cov_ratio should be larger than 0."

    negative_approx_kl = log_prob - old_log_prob
    abs_kl = negative_approx_kl.abs()
    ratio = torch.exp(negative_approx_kl)
    ppo_kl_abs = verl_F.masked_mean(negative_approx_kl.abs(), response_mask)
    pg_losses1 = -advantages * ratio
    pg_losses_kl = -advantages * ratio + ppo_kl_coef * abs_kl
    pg_losses = pg_losses1

    all_valid = response_mask > 0
    all_valid_idx = torch.nonzero(all_valid.reshape(-1), as_tuple=True)[0]
    all_valid_adv = advantages[all_valid].detach().reshape(-1).cpu()
    all_valid_logp = log_prob[all_valid].detach().reshape(-1).cpu()

    k = min(kl_cov_ratio, len(all_valid_adv))

    if k != 0:
        cov_lst_all = (all_valid_adv - all_valid_adv.mean()) * (all_valid_logp - all_valid_logp.mean())
        k_percent_nums = max(1, int(len(cov_lst_all) * kl_cov_ratio))
        large_cov_idxs = torch.topk(cov_lst_all, k_percent_nums, largest=True).indices

        if len(large_cov_idxs) != 0:
            large_cov_idxs = all_valid_idx[large_cov_idxs]
            pg_losses[large_cov_idxs // advantages.shape[1], large_cov_idxs % advantages.shape[1]] = pg_losses_kl[
                large_cov_idxs // advantages.shape[1], large_cov_idxs % advantages.shape[1]
            ]

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )
    pg_metrics = {
        "actor/ppo_kl": ppo_kl_abs.detach().item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("geo_mean")
def compute_policy_loss_geo_mean(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for GMPO.

    Adapted from paper https://arxiv.org/abs/2507.20673
    https://github.com/callsys/GMPO/blob/main/train_zero_math_gmpo.py

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            not used
    """

    assert config is not None
    assert not isinstance(config, AlgoConfig)
    clip_ratio = config.clip_ratio  # Clipping parameter. See https://arxiv.org/abs/1707.06347.
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio

    cliprange = clip_ratio
    cliprange_low = clip_ratio_low
    cliprange_high = clip_ratio_high
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability (uncomment it if you like)
    # negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # Clipping at token-level & Clipping wider
    sgn_advantage = torch.sign(advantages)
    negative_approx_kl_clamp = torch.clamp(negative_approx_kl, -cliprange_low, cliprange_high)
    negative_approx_kl_min = torch.min(sgn_advantage * negative_approx_kl, sgn_advantage * negative_approx_kl_clamp)
    negative_approx_kl_min = sgn_advantage * negative_approx_kl_min

    # Geometric-Mean Policy Optimization
    response_mask_sum = response_mask.sum(dim=-1)
    ratio = torch.exp((negative_approx_kl_min * response_mask).sum(dim=-1) / (response_mask_sum + 1e-8))
    # we only support sequence level advantage for now,
    # otherwise, below would be not consistent with the paper
    advantage = (advantages * response_mask).sum(dim=-1) / (response_mask_sum + 1e-8)
    pg_losses = -advantage * ratio

    # Apply rollout correction weights if provided
    # For geo_mean, IS weights are 2D (batch_size, seq_length) and need to be aggregated to sequence level
    if rollout_is_weights is not None:
        # Aggregate token-level weights to sequence level using geometric mean for consistency
        # Note: rollout_is_weights is always 2D regardless of aggregation mode
        seq_is_weights = torch.exp(
            (torch.log(rollout_is_weights + 1e-10) * response_mask).sum(dim=-1) / (response_mask_sum + 1e-8)
        )
        pg_losses = pg_losses * seq_is_weights

    pg_loss = torch.mean(pg_losses)

    # higher: ratio is too large that need clamp to clip_high (when adv > 0)
    clipped = torch.ne(negative_approx_kl, negative_approx_kl_clamp)
    pg_clipfrac = verl_F.masked_mean((clipped * (advantages > 0)).float(), response_mask)
    pg_clipfrac_lower = verl_F.masked_mean((clipped * (advantages < 0)).float(), response_mask)
    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("cispo")
def compute_policy_loss_cispo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig | ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for CISPO.

    See https://arxiv.org/pdf/2506.13585 for more details.
    """

    assert config is not None
    assert isinstance(config, ActorConfig)
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else config.clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else config.clip_ratio

    # Compute importance sampling ratio: π_θ / π_θ_old
    negative_approx_kl = log_prob - old_log_prob
    # Clamp for numerical stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # CISPO: Clip the importance sampling weights
    # KEY: Apply stop gradient to the clipped ratio
    # This prevents gradients from flowing through the ratio computation and clipping
    # Gradients only flow through log_prob in the final loss term
    clipped_ratio = torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    clipped_ratio_sg = clipped_ratio.detach()

    # CISPO objective function (to maximize): J = sg(clip(ratio)) * A * log π_θ
    # Loss function (to minimize): L = -J = -sg(clip(ratio)) * A * log_prob
    pg_losses = -clipped_ratio_sg * advantages * log_prob

    # Track clipping statistics
    pg_clipfrac = verl_F.masked_mean((ratio != clipped_ratio).float(), response_mask)

    # Apply rollout importance sampling weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )

    # For compatibility, return zero for pg_clipfrac_lower (not used in CISPO)
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)

    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    return pg_loss, pg_metrics


def compute_entropy_loss(logits, response_mask, loss_agg_mode: str = "token-mean"):
    """Compute categorical entropy loss (For backward compatibility)

    Args:
        logits (torch.Tensor): shape is (bs, response_length, vocab_size)
        response_mask (torch.Tensor): shape is (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    token_entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = agg_loss(loss_mat=token_entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    return entropy_loss


def compute_value_loss(
    vpreds: torch.Tensor,
    returns: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    cliprange_value: float,
    loss_agg_mode: str = "token-mean",
):
    """
    Compute the clipped value-function loss for PPO.

    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (torch.FloatTensor):
            Predicted values from the value head, shape (batch_size, response_length).
        values (torch.FloatTensor):
            Old (baseline) values from the value head, shape (batch_size, response_length).
        returns (torch.FloatTensor):
            Ground-truth returns, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the value loss calculation.
        cliprange_value (float):
            Clip range for value prediction updates.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".

    Returns:
        vf_loss (torch.FloatTensor):
            A scalar tensor containing the aggregated value-function loss.
        vf_clipfrac (float):
            Fraction of elements where the clipped loss was used.
    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    clipped_vf_losses = torch.max(vf_losses1, vf_losses2)
    vf_loss = 0.5 * agg_loss(loss_mat=clipped_vf_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), response_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob. Optionally using straight through to bind k2 on other
    kl penalty compute method for unbiased KL gradient estimation.
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:
        kl_estimate
    """
    forward_score = kl_penalty_forward(logprob, ref_logprob, kl_penalty)
    if not kl_penalty.endswith("+") or kl_penalty in ("mse", "k2"):
        return forward_score

    """
    The expectation of k1 and k3 estimator is the expectaed value of KL, but the expected gradient of k1 and k3
    estimator is not the expectaed gradient of KL. On the other hand k2 estimator gives right gradient estimator,
    so we use a straight through trick here if the kl_penalty method ends with '+', .e.g., k3+.
    """
    backward_score = 0.5 * (logprob - ref_logprob).square()

    return backward_score - backward_score.detach() + forward_score.detach()


def kl_penalty_forward(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:
        kl_estimate
    """
    if kl_penalty in ("kl", "k1"):
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty in ("low_var_kl", "k3"):
        kl = ref_logprob - logprob
        # For numerical stability
        kl = torch.clamp(kl, min=-20, max=20)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError


def compute_pf_ppo_reweight_data(
    data,
    reweight_method: str = "pow",
    weight_pow: float = 2.0,
):
    """Reweight the data based on the token_level_scores.

    Args:
        data: DataProto object, containing batch, non_tensor_batch and meta_info
        reweight_method: str, choices: "pow", "max_min", "max_random"
        weight_pow: float, the power of the weight

    Returns:

    """

    @torch.no_grad()
    def compute_weights(scores: torch.Tensor, reweight_method: str, weight_pow: float) -> torch.Tensor:
        """Compute importance weights for resampling based on scores.

        Args:
            scores (torch.Tensor): Tensor of scores to compute weights from.
            reweight_method (str): Method for computing weights ('pow', 'max_min', 'max_random').
            weight_pow (float): Power exponent for 'pow' method.

        Returns:
            torch.Tensor: Computed importance weights.

        Raises:
            ValueError: If reweight_method is not supported.
        """
        if reweight_method == "pow":
            weights = torch.pow(torch.abs(scores), weight_pow)
        elif reweight_method == "max_min":
            max_score = torch.max(scores)
            min_score = torch.min(scores)
            weights = torch.where((scores == max_score) | (scores == min_score), 1.0, 0.0)
        elif reweight_method == "max_random":
            max_score = torch.max(scores)
            weights = torch.where(scores == max_score, 0.4, 0.1)
        else:
            raise ValueError(f"Unsupported reweight_method: {reweight_method}")
        return weights

    scores = data.batch["token_level_scores"].sum(dim=-1)
    weights = compute_weights(scores, reweight_method, weight_pow)
    weights = torch.clamp(weights + 1e-8, min=1e-8)

    batch_size = scores.shape[0]
    sample_indices = torch.multinomial(weights, batch_size, replacement=True)

    resampled_batch = {key: tensor[sample_indices] for key, tensor in data.batch.items()}

    sample_indices_np = sample_indices.numpy()
    resampled_non_tensor_batch = {}
    for key, array in data.non_tensor_batch.items():
        if isinstance(array, np.ndarray):
            resampled_non_tensor_batch[key] = array[sample_indices_np]
        else:
            resampled_non_tensor_batch[key] = [array[i] for i in sample_indices_np]

    resampled_meta_info = {}
    for key, value in data.meta_info.items():
        if isinstance(value, list) and len(value) == batch_size:
            resampled_meta_info[key] = [value[i] for i in sample_indices_np]
        else:
            resampled_meta_info[key] = value

    from copy import deepcopy

    resampled_data = deepcopy(data)
    resampled_data.batch = type(data.batch)(resampled_batch)
    resampled_data.batch.batch_size = data.batch.batch_size
    resampled_data.non_tensor_batch = resampled_non_tensor_batch
    resampled_data.meta_info = resampled_meta_info

    return resampled_data


def compute_policy_loss_reinforce(
    rollout_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-sum",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute REINFORCE-style policy gradient loss with optional IS correction.

    This function implements policy gradient (REINFORCE) with optional importance
    sampling correction for rollout-training policy mismatch.

    Mathematical formulation:
        Without IS (rollout_is_weights=None):
            L = -E[log π(a|s) * A(s,a)]
            Gradient: ∇_θ L = -E[∇log π(a|s) * A] (standard REINFORCE)

        With IS (rollout_is_weights provided):
            L = -E_π_rollout[w * log π(a|s) * A(s,a)]
            where w = π_current / π_rollout (truncated IS weight)
            Gradient: ∇_θ L = -E[w * ∇log π(a|s) * A] (IS-corrected policy gradient)

    Args:
        rollout_log_prob: Log probabilities from rollout policy (e.g., vLLM BF16).
            Shape: (batch_size, seq_length). Used for KL computation.
        log_prob: Log probabilities from current training policy.
            Shape: (batch_size, seq_length)
        advantages: Advantage estimates for each token.
            Shape: (batch_size, seq_length)
        response_mask: Mask indicating valid tokens (1 for valid, 0 for padding).
            Shape: (batch_size, seq_length). Should already include rejection sampling.
        loss_agg_mode: Loss aggregation strategy (see agg_loss for details).
        config: Actor config (required for global_batch_info).
        rollout_is_weights: Pre-computed IS weights (π_current / π_rollout).
            Shape: (batch_size, seq_length). None to disable IS correction.

    Returns:
        Tuple of (loss, metrics):
            loss: Scalar policy gradient loss
            metrics: Dictionary with "actor/ppo_kl"

    Note:
        Unlike PPO (compute_policy_loss_vanilla), this function:
        - Does NOT use PPO clipping
        - Uses log π(a|s) directly (not ratio)
        - IS weights are applied as multiplicative factor
    """
    assert config is not None, "ActorConfig must be provided for REINFORCE loss"

    # Compute pure policy gradient loss with optional IS correction
    # Standard REINFORCE: L = -E[log π(a|s) * A]
    # With IS: L = -E[w * log π(a|s) * A] where w = π_current / π_rollout
    if rollout_is_weights is not None:
        # IS-corrected policy gradient: L = -E[stopgrad(w) · log π · A]
        pg_losses = -advantages * log_prob * rollout_is_weights
    else:
        # Standard REINFORCE: L = -E[log π · A]
        pg_losses = -advantages * log_prob

    # Aggregate loss
    pg_loss = agg_loss(
        loss_mat=pg_losses,
        loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        **config.global_batch_info,
    )

    # Compute KL divergence between current and rollout policy
    negative_approx_kl = log_prob - rollout_log_prob
    kl_divergence = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_metrics = {
        "actor/ppo_kl": kl_divergence.detach().item(),
    }

    return pg_loss, pg_metrics


@register_policy_loss("bypass_mode")
def compute_policy_loss_bypass_mode(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Bypass mode policy loss supporting both REINFORCE and PPO-clip.

    This function is the entry point for bypass mode, where old_log_prob = rollout_log_prob.
    It computes IS weights and rejection masks, then dispatches to either REINFORCE or
    PPO-clip loss based on the loss_type configuration.

    IMPORTANT - Bypass mode semantics:
        In bypass mode, the trainer sets old_log_prob = rollout_log_prob.
        This means:
        - For REINFORCE: We use IS weights w = π_current / π_rollout explicitly
        - For PPO-clip: The PPO ratio π_current / π_old = π_current / π_rollout
          already incorporates the IS correction through clipping, so we do NOT
          apply additional IS weights (would be double-counting)

    Loss types:
        - "ppo_clip" (default): PPO clipped objective (compute_policy_loss_vanilla)
            L = -E[min(r*A, clip(r)*A)] where r = π_current / π_rollout
            Note: IS weights are NOT applied (clipping handles the ratio)
        - "reinforce": REINFORCE-style policy gradient with IS correction
            L = -E[w * log π(a|s) * A] where w = π_current / π_rollout

    Args:
        old_log_prob: In bypass mode, this is actually rollout_log_prob.
            Shape: (batch_size, seq_length)
        log_prob: Current policy log probabilities.
            Shape: (batch_size, seq_length)
        advantages: Advantage estimates.
            Shape: (batch_size, seq_length)
        response_mask: Valid token mask (1=valid, 0=padding).
            Shape: (batch_size, seq_length)
        loss_agg_mode: Loss aggregation mode (passed to underlying loss function).
        config: Actor config containing rollout_correction settings in policy_loss.
        rollout_is_weights: Pre-computed IS weights (ignored, computed internally).

    Config options (in config.policy_loss.rollout_correction):
        loss_type: "ppo_clip" (default) or "reinforce"
        rollout_is: IS aggregation level ("token", "sequence", or None)
        rollout_is_threshold: Upper threshold for truncating IS weights (default: 2.0)
        rollout_rs: Rejection sampling level (see rollout_corr_helper for supported modes)
        rollout_rs_threshold: Threshold specification for rejection sampling
        rollout_is_batch_normalize: Whether to normalize IS weights to mean=1.0

    Returns:
        Tuple of (loss, metrics):
            loss: Scalar policy loss
            metrics: Dictionary with rollout correction metrics and actor/ppo_kl
    """
    from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_rejection_mask

    assert config is not None, "config is required for bypass_mode loss"

    # Extract rollout_correction config from policy_loss
    rollout_corr_config = config.policy_loss.get("rollout_correction", None) if hasattr(config, "policy_loss") else None

    if rollout_corr_config is None:
        raise ValueError(
            "rollout_correction config not found in policy_loss. "
            "When using loss_mode='bypass_mode', ensure rollout_correction config is passed."
        )

    # Extract parameters
    loss_type = rollout_corr_config.get("loss_type", "ppo_clip")
    rollout_is = rollout_corr_config.get("rollout_is", None)
    rollout_is_threshold = rollout_corr_config.get("rollout_is_threshold", 2.0)
    rollout_is_batch_normalize = rollout_corr_config.get("rollout_is_batch_normalize", False)
    rollout_rs = rollout_corr_config.get("rollout_rs", None)
    rollout_rs_threshold = rollout_corr_config.get("rollout_rs_threshold", None)

    # In bypass mode: old_log_prob IS rollout_log_prob
    rollout_log_prob = old_log_prob

    # Compute IS weights and rejection mask
    # Note: For PPO-clip, we still compute IS weights for metrics, but don't apply them
    with torch.no_grad():
        rollout_is_weights_proto, modified_response_mask, rollout_metrics = (
            compute_rollout_correction_and_rejection_mask(
                old_log_prob=log_prob,  # Current policy (for IS ratio: π_current / π_rollout)
                rollout_log_prob=rollout_log_prob,  # Rollout policy
                response_mask=response_mask,
                rollout_is=rollout_is,
                rollout_is_threshold=rollout_is_threshold,
                rollout_is_batch_normalize=rollout_is_batch_normalize,
                rollout_rs=rollout_rs,
                rollout_rs_threshold=rollout_rs_threshold,
            )
        )

    # Extract IS weights tensor (or None if disabled)
    computed_is_weights = rollout_is_weights_proto.batch["rollout_is_weights"] if rollout_is_weights_proto else None

    # Apply rejection mask (RS + veto)
    effective_mask = modified_response_mask

    # Dispatch to appropriate loss function based on loss_type
    if loss_type == "reinforce":
        # REINFORCE: Apply IS weights explicitly
        pg_loss, pg_metrics = compute_policy_loss_reinforce(
            rollout_log_prob=rollout_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=effective_mask,
            loss_agg_mode=loss_agg_mode,
            config=config,
            rollout_is_weights=computed_is_weights,
        )

    elif loss_type == "ppo_clip":
        # PPO-clip: The ratio π_current/π_old = π_current/π_rollout already handles IS
        # DO NOT apply IS weights - would be double-counting!
        # The clipping mechanism constrains the effective IS ratio
        pg_loss, pg_metrics = compute_policy_loss_vanilla(  # type: ignore[call-arg]
            old_log_prob=rollout_log_prob,  # = old_log_prob in bypass mode
            log_prob=log_prob,
            advantages=advantages,
            response_mask=effective_mask,
            loss_agg_mode=loss_agg_mode,
            config=config,
            rollout_is_weights=None,  # Explicitly None - no IS weights for PPO-clip
        )

    else:
        raise ValueError(f"Invalid loss_type: {loss_type}. Must be 'reinforce' or 'ppo_clip'.")

    # Merge rollout correction metrics
    pg_metrics.update(rollout_metrics)

    return pg_loss, pg_metrics
