# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from dataclasses import dataclass, field
from typing import Any, Optional

from omegaconf import MISSING

from verl.base_config import BaseConfig
from verl.trainer.config import CheckpointConfig
from verl.utils.profiler.config import ProfilerConfig

from .engine import FSDPEngineConfig, McoreEngineConfig
from .model import HFModelConfig
from .optimizer import OptimizerConfig

__all__ = [
    "SelfDistillationConfig",
    "PolicyLossConfig",
    "RouterReplayConfig",
    "ActorConfig",
    "FSDPActorConfig",
    "McoreActorConfig",
]


@dataclass
class SelfDistillationConfig(BaseConfig):
    """Configuration for self-distillation loss.

    Args:
        Distillation is enabled when policy_loss.loss_mode is "sdpo" or "sipo".
        full_logit_distillation (bool): Whether to use full-logit KL distillation.
        alpha (float): KL interpolation coefficient. 0.0=forward KL, 1.0=reverse KL, in-between=JSD.
        beta (float): Weight applied to the SIPO divergence term.
        success_reward_threshold (float): Minimum sequence reward to be considered successful.
        teacher_regularization (str): Teacher regularization mode. Options: "ema", "trust-region", "student".
        teacher_update_rate (float): EMA update rate for teacher weights, or trust-region mixing coefficient.
        distillation_topk (Optional[int]): If set, use top-k logits for distillation.
        distillation_add_tail (bool): Whether to add a tail bucket for top-k distillation.
        max_reprompt_len (int): Maximum length of the reprompted prompt.
        reprompt_truncation (str): Truncation method for the reprompted prompt (recommended to use "right" or "error").
        dont_reprompt_on_self_success (bool): Whether to not reprompt on self-success.
        remove_thinking_from_demonstration (bool): Whether to remove <think>...</think> tags from successful demonstrations before reprompting.
        student_reasoning_suffix (str): Student prompt suffix appended during preprocessing; stripped from teacher reprompt input.
        is_clip (Optional[float]): Clip value for distillation IS ratio; None disables IS weighting.
        reinforce_advantage (bool): RLSD (arXiv:2604.03128). Drop the divergence term and let
            the teacher reweight the reward advantage within each rollout:
            A_t = A_r * ((1 - lam) + lam * clip(exp(sign(A_r) * A_d), 1 - sipo_eps, 1 + sipo_eps)),
            A_d = log pi_teacher(y_t | x, r) - log pi_student(y_t | x). Requires alpha=1.0 and
            full_logit_distillation=False; the only distillation form that works with fused kernels.
        sipo_eps (float): bound on the per-token credit deviation (paper: 0.2).
        sipo_lambda_start / sipo_lambda_steps: lam decays linearly from the start value to 0 over
            this many steps, after which the update is plain GRPO (paper: 0.5 over 50).
        sipo_evidence ("opd" | "contrastive"): "opd" is RLSD as published, e_t = log pi_T(y_t|x, gold)
            - log pi_S(y_t|x), whose reweighting is lam*|A_r| times the on-policy-distillation
            gradient toward the answer-conditioned model ("act as if you knew the answer").
            "contrastive" uses e_t = log pi_T(y_t|x, gold) - log pi_T(y_t|x, wrong answer), the
            wrong answer being the group's most common wrong final answer: both sides are
            answer-conditioned, the decisiveness prior cancels, and e_t is nonzero only on tokens
            specific to the right or the wrong answer. Costs a second teacher forward pass.
        sipo_baseline ("none" | "confidence"): "confidence" replaces the raw log-ratio A_d with
            A_d minus its mean over tokens of the same student confidence (EMA-smoothed across
            micro batches at sipo_baseline_ema). The raw log-ratio has expectation -KL_t, most
            negative where the student is least sure, so under sign(A_r) it is a push against
            exploration whatever the outcome; the centred evidence is zero-mean at every
            confidence level and carries only token-specific information.
        sipo_negative_only (bool): reweight wrong rollouts only (reference repo flag).
        sipo_all_wrong_kappa (float): > 0 gives the rollouts of an all-wrong group (A_r = 0, so the
            multiplicative reweighting leaves them at 0) the additive contrastive credit
            A_t = sipo_all_wrong_kappa * clip(e_t, +-sipo_eps), a constant. A wrong rollout in a
            mixed group gets lam*|A_r|*e_t from the teacher (first order) and all_wrong/c logs the
            mean |A_r| of those rollouts, so lam*c (about 0.15 early on) is the same teacher term.
            Needs sipo_evidence=contrastive. Mixed groups keep the multiplicative form; all-correct
            groups stay at 0.
        sipo_contrast_negative ("own" | "mode"): which wrong answer the second teacher is shown
            under sipo_evidence=contrastive. "own": a wrong rollout is contrasted against its own
            parseable final answer (the matched distractor); correct, truncated or gold-equal rows
            use the group's most common wrong answer. "mode": every row uses the group mode.
        sipo_contrast_context ("answer" | "trajectory"): what the two teachers are conditioned on.
            "answer": the gold answer vs a wrong answer in the gold block (math, gpqa/sciknoweval,
            tooluse; the ground truth is rendered in the scorer's own "pred" format so the two
            prompts differ only in content). "trajectory": a successful sibling rollout vs a failed
            sibling rollout under the same solution_template, each excluding the row itself, the
            gold block unused; for tasks with no answer to show, such as code (the ground truth is
            a test suite). Rows lacking either sibling get no contrast. "auto": the trajectory
            pair where both siblings exist, else the answer pair, so all-wrong groups keep their
            contrast (gold vs their own wrong answers); needs gold_answer_for_all for the fallback.
        sipo_contrast_unjudged ("skip" | "mode"): a WRONG rollout without a valid judged answer
            of its own (truncated, outside the verifier's window, malformed) has no matched negative.
            "skip": no teacher term for it; "mode": fall back to the group's most common wrong answer
            (the behaviour of the runs before 2026-09-23). Correct rollouts always use the mode.
        sipo_form ("multiplicative" | "additive"): how the evidence enters the advantage.
            "multiplicative" is RLSD/CEPO, A_t = A_r*((1-lam)+lam*clip(exp(sign(A_r)*e_t))): the
            teacher term scales with |A_r| (first order lam*|A_r|*e_t) and is 0 on all-wrong
            groups unless sipo_all_wrong_kappa adds it. "additive" is RLCSD's form for EVERY row,
            A_t = A_r + sipo_kappa*clip(e_t, +-sipo_eps): the same constant on a 1-of-8 group's
            wrong rollouts and on an all-wrong group, with a sign-preserving clamp on mixed rows
            (never engaged while sipo_kappa*sipo_eps < 1/G). Needs sipo_evidence=contrastive and
            sipo_all_wrong_kappa=0; lam is unused.
        sipo_kappa (float): the constant of the additive form.
        gold_answer_for_all (bool): every row's teacher sees the dataset's final answer as its
            privileged context r (RLSD's setup). Otherwise the SIPO context, a successful sibling.
        teacher_sync_every (int): for teacher_regularization="periodic", copy the student into the
            teacher every N steps and freeze it in between (paper: 10).
        debug_print_every (int): every N steps rank 0 prints one row of the reweighting token by
            token, with the formula recomputed next to the values (0 = off).
        impute_group_advantage (bool): give all-wrong groups A = -impute_wrong_scale * c (c = batch
            mean |A| of the wrong rollouts in mixed groups) and all-correct groups
            +impute_correct_scale * c, instead of GRPO's 0. The teacher then reweights ONLY those
            rows; mixed groups stay bit-identical to GRPO.
        include_incorrect_attempt (bool): Give a SUCCESSFUL rollout a sibling's incorrect attempt as
            teacher context. False leaves it without one, as in the SIPO repo.
        reprompt_template (str): Template for reprompting. Uses {prompt}, {solution}, {feedback}, {incorrect_attempt} placeholders.
        solution_template (str): Template for formatting solution section. Uses {successful_previous_attempt} placeholder.
        feedback_template (str): Template for formatting feedback section. Uses {feedback_raw} placeholder.
        incorrect_attempt_template (str): Template for formatting an incorrect peer attempt.
        include_environment_feedback (bool): Whether to include environment feedback in reprompting for wrong attempts.
        environment_feedback_only_without_solution (bool): If True, only use feedback when no solution is available (ignore feedback when solution exists).
        reprompt_template_feedback (str): Template for reprompting with feedback but no solution.
        reprompt_template_feedback_solution (str): Template for reprompting with both feedback and solution.
    """

    full_logit_distillation: bool = True
    alpha: float = 0.0
    beta: float = 1.0
    success_reward_threshold: float = 1.0
    teacher_regularization: str = "ema"
    teacher_update_rate: float = 0.05
    distillation_topk: Optional[int] = None
    distillation_add_tail: bool = True
    max_reprompt_len: int = 10240
    reprompt_truncation: str = "right"
    dont_reprompt_on_self_success: bool = False
    remove_thinking_from_demonstration: bool = False
    include_incorrect_attempt: bool = True
    is_clip: Optional[float] = None
    reinforce_advantage: bool = False
    sipo_eps: float = 0.2
    sipo_lambda_start: float = 0.5
    sipo_lambda_steps: int = 50
    sipo_evidence: str = "opd"
    sipo_baseline: str = "none"
    sipo_baseline_ema: float = 0.1
    sipo_negative_only: bool = False
    sipo_all_wrong_kappa: float = 0.0
    sipo_contrast_negative: str = "own"
    sipo_contrast_context: str = "answer"
    sipo_contrast_unjudged: str = "skip"
    sipo_form: str = "multiplicative"
    sipo_kappa: float = 0.0
    gold_answer_for_all: bool = False
    gold_answer_template: str = (
        "Here is a reference solution:\n{gold_answer}\n\n"
        "After understanding the reference solution, please try to solve this problem "
        "using your own approach below.\n\n---\n\n"
    )
    teacher_sync_every: int = 10
    debug_print_every: int = 1
    impute_group_advantage: bool = False
    impute_wrong_scale: float = 1.0
    impute_correct_scale: float = 0.0
    student_reasoning_suffix: str = "\n\nPlease reason step by step before generating your final answer."
    reprompt_template: str = (
        "{prompt}{solution}{incorrect_attempt}{feedback}"
        "\n\nUse the context above to solve the original question correctly. If an incorrect attempt or feedback is "
        "shown, avoid the identified error. Please reason step by step before generating your final answer."
    )
    solution_template: str = (
        "\n\n# Successful reference solution:"
        "\n\n{successful_previous_attempt}"
    )
    incorrect_attempt_template: str = (
        "\n\n# Incorrect reference attempt:"
        "\n\n{incorrect_attempt}"
    )
    feedback_template: str = (
        "\n\n# Environment feedback for the incorrect reference attempt:"
        "\n\n{feedback_raw}"
    )
    include_environment_feedback: bool = False
    environment_feedback_only_without_solution: bool = False

    def __post_init__(self):
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"self_distillation.alpha must be in [0,1], got {self.alpha}")
        beta = float(self.beta)
        if beta < 0.0:
            raise ValueError(f"self_distillation.beta must be non-negative, got {self.beta}")
        valid_teacher_regularization = ["ema", "trust-region", "student", "periodic"]
        if self.teacher_regularization not in valid_teacher_regularization:
            raise ValueError(
                "self_distillation.teacher_regularization must be one of "
                f"{valid_teacher_regularization}, got {self.teacher_regularization}"
            )
        if not 0.0 <= self.teacher_update_rate <= 1.0:
            raise ValueError(
                f"self_distillation.teacher_update_rate must be in [0,1], got {self.teacher_update_rate}"
            )
        if self.distillation_topk is not None and self.distillation_topk <= 0:
            raise ValueError(
                f"self_distillation.distillation_topk must be a positive integer, got {self.distillation_topk}"
            )
        if self.is_clip is not None and self.is_clip <= 0:
            raise ValueError(f"self_distillation.is_clip must be positive, got {self.is_clip}")
        if self.reinforce_advantage:
            if self.alpha != 1.0:
                raise ValueError(
                    "self_distillation.reinforce_advantage requires alpha=1.0 (reverse KL), "
                    f"got {self.alpha}"
                )
            if self.full_logit_distillation:
                raise ValueError(
                    "self_distillation.reinforce_advantage requires full_logit_distillation=False"
                )
        if not 0.0 < self.sipo_eps < 1.0:
            raise ValueError(f"self_distillation.sipo_eps must be in (0, 1), got {self.sipo_eps}")
        if not 0.0 <= self.sipo_lambda_start <= 1.0:
            raise ValueError(f"self_distillation.sipo_lambda_start must be in [0, 1], got {self.sipo_lambda_start}")
        if self.sipo_evidence not in ("opd", "contrastive"):
            raise ValueError(f"self_distillation.sipo_evidence must be 'opd' or 'contrastive', got {self.sipo_evidence!r}")
        if self.sipo_baseline not in ("none", "confidence"):
            raise ValueError(f"self_distillation.sipo_baseline must be 'none' or 'confidence', got {self.sipo_baseline!r}")
        if not 0.0 < self.sipo_baseline_ema <= 1.0:
            raise ValueError(f"self_distillation.sipo_baseline_ema must be in (0, 1], got {self.sipo_baseline_ema}")
        if self.sipo_contrast_negative not in ("own", "mode"):
            raise ValueError(f"self_distillation.sipo_contrast_negative must be 'own' or 'mode', got {self.sipo_contrast_negative!r}")
        if self.sipo_contrast_context not in ("answer", "trajectory", "auto"):
            raise ValueError(f"self_distillation.sipo_contrast_context must be 'answer', 'trajectory' or 'auto', got {self.sipo_contrast_context!r}")
        if self.sipo_contrast_context != "answer" and self.sipo_evidence != "contrastive":
            raise ValueError(f"self_distillation.sipo_contrast_context={self.sipo_contrast_context!r} needs sipo_evidence='contrastive'")
        if self.sipo_contrast_unjudged not in ("skip", "mode"):
            raise ValueError(f"self_distillation.sipo_contrast_unjudged must be 'skip' or 'mode', got {self.sipo_contrast_unjudged!r}")
        if self.sipo_form not in ("multiplicative", "additive"):
            raise ValueError(f"self_distillation.sipo_form must be 'multiplicative' or 'additive', got {self.sipo_form!r}")
        if self.sipo_kappa < 0.0:
            raise ValueError(f"self_distillation.sipo_kappa must be >= 0, got {self.sipo_kappa}")
        if self.sipo_form == "additive":
            if self.sipo_evidence != "contrastive":
                raise ValueError("self_distillation.sipo_form='additive' needs sipo_evidence='contrastive'")
            if not self.reinforce_advantage:
                raise ValueError("self_distillation.sipo_form='additive' needs reinforce_advantage=True")
            if self.sipo_kappa <= 0.0:
                raise ValueError("self_distillation.sipo_form='additive' needs sipo_kappa > 0")
            if self.sipo_all_wrong_kappa > 0.0:
                raise ValueError("self_distillation.sipo_form='additive' covers all-wrong groups itself; set sipo_all_wrong_kappa=0")
            if self.impute_group_advantage:
                raise ValueError("self_distillation.sipo_form='additive' cannot be combined with impute_group_advantage")
        if self.sipo_all_wrong_kappa < 0.0:
            raise ValueError(f"self_distillation.sipo_all_wrong_kappa must be >= 0, got {self.sipo_all_wrong_kappa}")
        if self.sipo_all_wrong_kappa > 0.0:
            if self.sipo_evidence != "contrastive":
                raise ValueError("self_distillation.sipo_all_wrong_kappa > 0 needs sipo_evidence='contrastive'")
            if not self.reinforce_advantage:
                raise ValueError("self_distillation.sipo_all_wrong_kappa > 0 needs reinforce_advantage=True")
            if self.impute_group_advantage:
                raise ValueError(
                    "self_distillation.sipo_all_wrong_kappa cannot be combined with impute_group_advantage "
                    "(imputed rows have A_r != 0 and already go through the multiplicative path)"
                )
        if self.sipo_lambda_steps < 0:
            raise ValueError(f"self_distillation.sipo_lambda_steps must be >= 0, got {self.sipo_lambda_steps}")
        if self.impute_wrong_scale < 0.0 or self.impute_correct_scale < 0.0:
            raise ValueError("self_distillation.impute_wrong_scale / impute_correct_scale must be >= 0")
        if self.teacher_sync_every < 1:
            raise ValueError(f"self_distillation.teacher_sync_every must be >= 1, got {self.teacher_sync_every}")


@dataclass
class RouterReplayConfig(BaseConfig):
    """Configuration for router replay in MoE models.

    This configuration controls the routing behavior for Mixture of Experts (MoE) models,
    allowing for deterministic training through route recording and replay.

    Args:
        mode (str): Router replay mode. Options: 'disabled', 'R2', 'R3'.
            - 'disabled': No router replay functionality
            - 'R2': Use Router Replay routing strategy
            - 'R3': Use Rollout Router Replay routing strategy
        record_file (Optional[str]): File path to save recorded routing decisions.
            Required when mode is 'record', 'R2', or 'R3'.
        replay_file (Optional[str]): File path to load recorded routing decisions for replay.
            Required when mode is 'replay'.
    """

    mode: str = "disabled"
    record_file: Optional[str] = None
    replay_file: Optional[str] = None

    def __post_init__(self):
        """Validate router replay configuration."""
        valid_modes = ["disabled", "R2", "R3"]
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid router_replay mode: {self.mode}. Must be one of {valid_modes}")


@dataclass
class PolicyLossConfig(BaseConfig):
    """Configuration for policy loss computation.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        loss_mode (str): Loss function mode. Options: 'vanilla', 'clip-cov', 'kl-cov', 'gpg', 'sdpo', 'sipo'.
        clip_cov_ratio (float): Ratio of tokens to be clipped for clip-cov loss.
        clip_cov_lb (float): Lower bound for clip-cov loss.
        clip_cov_ub (float): Upper bound for clip-cov loss.
        kl_cov_ratio (float): Ratio of tokens to be applied KL penalty for kl-cov loss.
        ppo_kl_coef (float): KL divergence penalty coefficient.
    """

    loss_mode: str = "vanilla"
    clip_cov_ratio: float = 0.0002
    clip_cov_lb: float = 1.0
    clip_cov_ub: float = 5.0
    kl_cov_ratio: float = 0.0002
    ppo_kl_coef: float = 0.1


@dataclass
class ActorConfig(BaseConfig):
    """Configuration for actor model training.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        strategy (str): Training strategy. Must be specified.
        ppo_mini_batch_size (int): Mini-batch size for PPO training.
        ppo_micro_batch_size (Optional[int]): Micro-batch size for PPO training.
            If None, uses ppo_micro_batch_size_per_gpu.
        ppo_micro_batch_size_per_gpu (Optional[int]): Micro-batch size per GPU for PPO training.
        use_dynamic_bsz (bool): Whether to use dynamic batch sizing.
        ppo_max_token_len_per_gpu (int): Maximum token length per GPU for PPO training.
        clip_ratio (float): PPO clipping ratio for policy loss.
        clip_ratio_low (float): Lower bound for PPO clipping ratio.
        clip_ratio_high (float): Upper bound for PPO clipping ratio.
        policy_loss (PolicyLossConfig): Configuration for policy loss computation.
        clip_ratio_c (float): Clipping ratio for critic loss.
        loss_agg_mode (str): Loss aggregation mode. Options: 'token-mean', 'sample-mean'.
        loss_scale_factor (Optional[int]): Scale factor for 'seq-mean-token-sum-norm' loss aggregation mode.
            If None, uses response_length. Set to a constant to ensure consistent normalization.
        entropy_coeff (float): Entropy coefficient for regularization.
        tau_pos (float): Positive tau for SAPO smoothing (>= 1.0 keeps rewards stable).
        tau_neg (float): Negative tau for SAPO smoothing (> tau_pos for asymmetry).
        use_kl_loss (bool): Whether to use KL divergence loss.
        use_torch_compile (bool): Whether to use torch.compile for optimization.
        kl_loss_coef (float): KL divergence loss coefficient.
        kl_loss_type (str): Type of KL loss to use.
        ppo_epochs (int): Number of PPO epochs per training step.
        shuffle (bool): Whether to shuffle data during training.
        checkpoint (CheckpointConfig): Configuration for checkpointing.
        optim (OptimizerConfig): Configuration for optimizer.
        use_fused_kernels (bool): Whether to use custom fused kernels (e.g., FlashAttention, fused MLP).
        data_loader_seed (int): Seed for data loader. If None, uses global seed.
        router_replay (RouterReplayConfig): Configuration for router replay in MoE models.
    """

    _mutable_fields = BaseConfig._mutable_fields | {
        "ppo_mini_batch_size",
        "ppo_micro_batch_size",
        "ppo_micro_batch_size_per_gpu",
        "ppo_infer_micro_batch_size_per_gpu",
        "engine",
        "model_config",
    }

    strategy: str = MISSING
    ppo_mini_batch_size: int = 256
    ppo_micro_batch_size: Optional[int] = None  # deprecate
    ppo_micro_batch_size_per_gpu: Optional[int] = None
    ppo_infer_micro_batch_size_per_gpu: Optional[int] = None
    use_dynamic_bsz: bool = False
    ppo_max_token_len_per_gpu: int = 16384
    ppo_infer_max_token_len_per_gpu: int = 16384
    clip_ratio: float = 0.2
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.2
    freeze_vision_tower: bool = False
    policy_loss: PolicyLossConfig = field(default_factory=PolicyLossConfig)
    clip_ratio_c: float = 3.0
    loss_agg_mode: str = "token-mean"
    loss_scale_factor: Optional[int] = None
    entropy_coeff: float = 0
    tau_pos: float = 1.0
    tau_neg: float = 1.05
    calculate_entropy: bool = False
    use_kl_loss: bool = False
    # Whether to enable PrefixGrouper-based shared-prefix forward
    use_prefix_grouper: bool = False
    use_torch_compile: bool = True
    kl_loss_coef: float = 0.001
    kl_loss_type: str = "low_var_kl"
    ppo_epochs: int = 1
    shuffle: bool = False
    data_loader_seed: int = 1
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    use_fused_kernels: bool = False
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    engine: BaseConfig = field(default_factory=BaseConfig)
    rollout_n: int = MISSING  # must be override by sampling config
    model_config: HFModelConfig = field(default_factory=BaseConfig)
    router_replay: RouterReplayConfig = field(default_factory=RouterReplayConfig)
    self_distillation: SelfDistillationConfig = field(default_factory=SelfDistillationConfig)

    # Store global batch info for loss aggregation:
    # dp_size: data parallel size
    # batch_num_tokens: number of valid tokens in global batch
    # global_batch_size: global batch size
    global_batch_info: dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate actor configuration parameters."""
        assert self.strategy != MISSING
        assert self.rollout_n != MISSING
        if not self.use_dynamic_bsz:
            if self.ppo_micro_batch_size is not None and self.ppo_micro_batch_size_per_gpu is not None:
                raise ValueError(
                    "[actor] You have set both 'actor.ppo_micro_batch_size' AND 'actor.ppo_micro_batch_size_per_gpu'. "
                    "Please remove 'actor.ppo_micro_batch_size' because only '*_ppo_micro_batch_size_per_gpu' is "
                    "supported (the former is deprecated)."
                )
            else:
                assert not (self.ppo_micro_batch_size is None and self.ppo_micro_batch_size_per_gpu is None), (
                    "[actor] Please set at least one of 'actor.ppo_micro_batch_size' or "
                    "'actor.ppo_micro_batch_size_per_gpu' if use_dynamic_bsz is not enabled."
                )

        valid_loss_agg_modes = [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ]
        if self.loss_agg_mode not in valid_loss_agg_modes:
            raise ValueError(f"Invalid loss_agg_mode: {self.loss_agg_mode}")

    def validate(self, n_gpus: int, train_batch_size: int, model_config: dict = None):
        """Validate actor configuration with runtime parameters."""
        if not self.use_dynamic_bsz:
            if train_batch_size < self.ppo_mini_batch_size:
                raise ValueError(
                    f"train_batch_size ({train_batch_size}) must be >= "
                    f"actor.ppo_mini_batch_size ({self.ppo_mini_batch_size})"
                )

            sp_size = getattr(self, "ulysses_sequence_parallel_size", 1)
            if self.ppo_micro_batch_size is not None:
                if self.ppo_mini_batch_size % self.ppo_micro_batch_size != 0:
                    raise ValueError(
                        f"ppo_mini_batch_size ({self.ppo_mini_batch_size}) must be divisible by "
                        f"ppo_micro_batch_size ({self.ppo_micro_batch_size})"
                    )
                if self.ppo_micro_batch_size * sp_size < n_gpus:
                    raise ValueError(
                        f"ppo_micro_batch_size ({self.ppo_micro_batch_size}) * "
                        f"ulysses_sequence_parallel_size ({sp_size}) must be >= n_gpus ({n_gpus})"
                    )

    @staticmethod
    def _check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
        """Validate mutually exclusive micro batch size configuration options."""
        param = "ppo_micro_batch_size"
        param_per_gpu = f"{param}_per_gpu"

        if mbs is None and mbs_per_gpu is None:
            raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

        if mbs is not None and mbs_per_gpu is not None:
            raise ValueError(
                f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
                f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
            )


@dataclass
class McoreActorConfig(ActorConfig):
    """Configuration for Megatron actor models.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        strategy (str): Training strategy set to 'megatron' for Megatron parallelism.
        load_weight (bool): Whether to load model weights from checkpoint.
        megatron (dict[str, Any]): Configuration for Megatron parallelism settings.
        profile (dict[str, Any]): Configuration for profiling settings.
    """

    strategy: str = "megatron"
    load_weight: bool = True
    megatron: McoreEngineConfig = field(default_factory=McoreEngineConfig)
    profile: dict[str, Any] = field(default_factory=dict)
    use_rollout_log_probs: bool = False

    def __post_init__(self):
        """Validate FSDP actor configuration parameters."""
        super().__post_init__()
        self.engine = self.megatron


@dataclass
class FSDPActorConfig(ActorConfig):
    """Configuration for FSDP actor models.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        strategy (str): Training strategy set to 'fsdp' for Fully Sharded Data Parallel.
        grad_clip (float): Gradient clipping threshold.
        ulysses_sequence_parallel_size (int): [DEPRECATED] Ulysses sequence parallel size for long sequences.
        entropy_from_logits_with_chunking (bool): Whether to compute entropy from logits
            with chunking for memory efficiency.
        entropy_checkpointing (bool): Whether to use gradient checkpointing for entropy computation.
        fsdp_config (dict[str, Any]): Configuration for FSDP settings.
        use_remove_padding (bool): Whether to remove padding tokens in inputs during training
    """

    strategy: str = "fsdp"
    grad_clip: float = 1.0
    ulysses_sequence_parallel_size: int = 1
    entropy_from_logits_with_chunking: bool = False
    entropy_checkpointing: bool = False
    fsdp_config: FSDPEngineConfig = field(default_factory=FSDPEngineConfig)
    use_remove_padding: bool = False
    use_rollout_log_probs: bool = False
    calculate_sum_pi_squared: bool = False
    sum_pi_squared_checkpointing: bool = False

    def __post_init__(self):
        """Validate FSDP actor configuration parameters."""
        super().__post_init__()
        self.engine = self.fsdp_config

        # backward compatibility
        if self.ulysses_sequence_parallel_size > 1:
            self.fsdp_config.ulysses_sequence_parallel_size = self.ulysses_sequence_parallel_size

    def validate(self, n_gpus: int, train_batch_size: int, model_config: dict = None):
        """Validate FSDP actor configuration with runtime parameters."""
        super().validate(n_gpus, train_batch_size, model_config)

        if self.strategy in {"fsdp", "fsdp2"} and self.ulysses_sequence_parallel_size > 1:
            if model_config and not model_config.get("use_remove_padding", False):
                raise ValueError(
                    "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."
                )
