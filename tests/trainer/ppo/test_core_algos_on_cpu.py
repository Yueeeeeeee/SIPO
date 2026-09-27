# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import random
import math
import unittest
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import verl.trainer.ppo.core_algos
from verl.trainer.ppo.core_algos import (
    compute_self_instruction_loss,
    compute_gae_advantage_return,
    compute_grpo_outcome_advantage,
    compute_grpo_vectorized_outcome_advantage,
    compute_rloo_outcome_advantage,
    compute_rloo_vectorized_outcome_advantage,
    get_adv_estimator_fn,
    register_adv_est,
)


def mock_test_fn():
    pass


class TestRegisterAdvEst(unittest.TestCase):
    def setUp(self):
        """Clear the registry before each test"""
        verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY.clear()
        verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY = {
            "gae": lambda x: x * 2,
            "vtrace": lambda x: x + 1,
        }
        self.ADV_ESTIMATOR_REGISTRY = verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY

    def tearDown(self) -> None:
        verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY.clear()
        return super().tearDown()

    def test_register_new_function(self):
        """Test registering a new function with a string name"""

        @register_adv_est("test_estimator")
        def test_fn():
            pass

        self.assertIn("test_estimator", self.ADV_ESTIMATOR_REGISTRY)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["test_estimator"], test_fn)

    def test_register_with_enum(self):
        """Test registering with an enum value (assuming AdvantageEstimator exists)"""
        from enum import Enum

        class AdvantageEstimator(Enum):
            TEST = "test_enum_estimator"

        @register_adv_est(AdvantageEstimator.TEST)
        def test_fn():
            pass

        self.assertIn("test_enum_estimator", self.ADV_ESTIMATOR_REGISTRY)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["test_enum_estimator"], test_fn)

    def test_duplicate_registration_same_function(self):
        """Test that registering the same function twice doesn't raise an error"""
        register_adv_est("duplicate_test")(mock_test_fn)
        register_adv_est("duplicate_test")(mock_test_fn)

        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["duplicate_test"], mock_test_fn)

    def test_duplicate_registration_different_function(self):
        """Test that registering different functions with same name raises ValueError"""

        @register_adv_est("conflict_test")
        def test_fn1():
            pass

        with self.assertRaises(ValueError):

            @register_adv_est("conflict_test")
            def test_fn2():
                pass

    def test_decorator_preserves_function(self):
        """Test that the decorator returns the original function"""

        def test_fn():
            return "original"

        decorated = register_adv_est("preserve_test")(test_fn)
        self.assertEqual(decorated(), "original")

    def test_multiple_registrations(self):
        """Test registering multiple different functions"""
        init_adv_count = len(self.ADV_ESTIMATOR_REGISTRY)

        @register_adv_est("estimator1")
        def fn1():
            pass

        @register_adv_est("estimator2")
        def fn2():
            pass

        self.assertEqual(len(self.ADV_ESTIMATOR_REGISTRY), 2 + init_adv_count)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["estimator1"], fn1)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["estimator2"], fn2)

    def test_get_adv_estimator_fn_valid_names(self):
        """Test that valid names return the correct function from registry."""
        # Test GAE
        gae_fn = get_adv_estimator_fn("gae")
        assert gae_fn(5) == 10  # 5 * 2 = 10

        # Test Vtrace
        vtrace_fn = get_adv_estimator_fn("vtrace")
        assert vtrace_fn(5) == 6  # 5 + 1 = 6

    def test_get_adv_estimator_fn_invalid_name(self):
        """Test that invalid names raise ValueError."""
        with pytest.raises(ValueError) as excinfo:
            get_adv_estimator_fn("invalid_name")
        assert "Unknown advantage estimator simply: invalid_name" in str(excinfo.value)

    def test_get_adv_estimator_fn_case_sensitive(self):
        """Test that name lookup is case-sensitive."""
        with pytest.raises(ValueError):
            get_adv_estimator_fn("GAE")  # Different case


def test_multi_turn_compute_gae_advantage_return():
    """Test multi-turn GAE skip observation tokens."""
    gamma = random.uniform(0.0, 1.0)
    lam = random.uniform(0.0, 1.0)

    rewards = torch.tensor([[0.0, 0.0, 0.1, 0.1, 0.1, 0.0, 0.0, 0.1, 1.0, 0.0, 0.0]], dtype=torch.float)

    values1 = torch.tensor(
        [
            [
                random.uniform(-100.0, 100.0),
                random.random(),
                4.0,
                5.0,
                6.0,
                random.uniform(-100.0, 0),
                random.random(),
                7.0,
                9.0,
                0.0,
                0.0,
            ]
        ],
        dtype=torch.float,
    )

    values2 = torch.tensor(
        [
            [
                random.random(),
                random.uniform(-100.0, 100.0),
                4.0,
                5.0,
                6.0,
                random.random(),
                random.uniform(0.0, 100.0),
                7.0,
                9.0,
                0.0,
                0.0,
            ]
        ],
        dtype=torch.float,
    )

    response_mask = torch.tensor([[0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0]], dtype=torch.float)

    adv1, ret1 = compute_gae_advantage_return(rewards, values1, response_mask, gamma, lam)
    adv2, ret2 = compute_gae_advantage_return(rewards, values2, response_mask, gamma, lam)

    ret1 *= response_mask
    ret2 *= response_mask
    assert torch.equal(adv1, adv2), f"{adv1=}, {adv2=}"
    assert torch.equal(ret1, ret2), f"{ret1=}, {ret2=}"
    print(f" [CORRECT] \n\n{adv1=}, \n\n{ret1=}")


def _make_group_index(batch_size: int, num_groups: int) -> np.ndarray:
    """Create a numpy index array ensuring each group has at least 2 samples."""
    assert num_groups * 2 <= batch_size, "batch_size must allow >=2 samples per group"
    counts: list[int] = [2] * num_groups
    remaining = batch_size - 2 * num_groups
    for _ in range(remaining):
        counts[random.randrange(num_groups)] += 1
    index = []
    for gid, c in enumerate(counts):
        index.extend([gid] * c)
    random.shuffle(index)
    return np.asarray(index, dtype=np.int64)


def _rand_mask(batch_size: int, seq_len: int) -> torch.Tensor:
    mask = torch.randint(0, 2, (batch_size, seq_len), dtype=torch.int64).float()
    rows_without_one = (mask.sum(dim=-1) == 0).nonzero(as_tuple=True)[0]
    if len(rows_without_one) > 0:
        mask[rows_without_one, -1] = 1.0
    return mask


@pytest.mark.parametrize(
    "batch_size,seq_len,num_groups,seed",
    [
        (64, 128, 5, 0),
        (128, 256, 8, 1),
        (512, 512, 10, 2),
    ],
)
def test_rloo_and_vectorized_equivalence(batch_size: int, seq_len: int, num_groups: int, seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    index = _make_group_index(batch_size, num_groups)
    response_mask = _rand_mask(batch_size, seq_len)
    base_rewards = torch.randn(batch_size, seq_len, dtype=torch.float32)
    token_level_rewards = base_rewards * response_mask
    adv1, ret1 = compute_rloo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )
    adv2, ret2 = compute_rloo_vectorized_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )
    # Print concise diagnostics for visibility during test runs
    adv_max_diff = (adv1 - adv2).abs().max().item()
    ret_max_diff = (ret1 - ret2).abs().max().item()
    total_mask_tokens = int(response_mask.sum().item())
    print(
        f"[RLOO] seed={seed} groups={num_groups} shape={adv1.shape} "
        f"mask_tokens={total_mask_tokens} adv_max_diff={adv_max_diff:.3e} ret_max_diff={ret_max_diff:.3e}"
    )
    assert adv1.shape == adv2.shape == (batch_size, seq_len)
    assert ret1.shape == ret2.shape == (batch_size, seq_len)
    assert torch.allclose(adv1, adv2, rtol=1e-5, atol=1e-6)
    assert torch.allclose(ret1, ret2, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "batch_size,seq_len,num_groups,seed",
    [
        (64, 128, 5, 0),
        (128, 256, 8, 1),
        (512, 512, 10, 2),
    ],
)
def test_grpo_and_vectorized_equivalence(batch_size: int, seq_len: int, num_groups: int, seed: int):
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Generate group indices (numpy array of shape [batch_size])
    index = _make_group_index(batch_size, num_groups)

    # Generate binary response mask (at least one valid token per row)
    response_mask = _rand_mask(batch_size, seq_len)

    # Generate token-level rewards and apply mask
    base_rewards = torch.randn(batch_size, seq_len, dtype=torch.float32)
    token_level_rewards = base_rewards * response_mask

    # Compute GRPO outcome advantage (original implementation)
    adv1, ret1 = compute_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )

    # Compute GRPO outcome advantage (vectorized implementation)
    adv2, ret2 = compute_grpo_vectorized_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )

    # Diagnostic info for visibility (same style as RLOO test)
    adv_max_diff = (adv1 - adv2).abs().max().item()
    ret_max_diff = (ret1 - ret2).abs().max().item()
    total_mask_tokens = int(response_mask.sum().item())
    print(
        f"[GRPO] seed={seed} groups={num_groups} shape={adv1.shape} "
        f"mask_tokens={total_mask_tokens} adv_max_diff={adv_max_diff:.3e} ret_max_diff={ret_max_diff:.3e}"
    )

    # Assert shape and numerical equivalence
    assert adv1.shape == adv2.shape == (batch_size, seq_len)
    assert ret1.shape == ret2.shape == (batch_size, seq_len)
    assert torch.allclose(adv1, adv2, rtol=1e-5, atol=1e-6)
    assert torch.allclose(ret1, ret2, rtol=1e-5, atol=1e-6)


def test_self_instruction_loss_directionality():
    student_log_probs = torch.tensor([[-2.0], [-2.0]], dtype=torch.float32, requires_grad=True)
    teacher_log_probs = torch.tensor([[-2.0], [-2.0]], dtype=torch.float32)
    response_mask = torch.ones_like(student_log_probs)
    advantages = torch.tensor([[1.0], [-1.0]], dtype=torch.float32)
    cfg = SimpleNamespace(
        is_clip=None,
        alpha=1.0,
        full_logit_distillation=False,
    )

    loss, metrics = compute_self_instruction_loss(
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        response_mask=response_mask,
        self_distillation_config=cfg,
        advantages=advantages,
        old_log_probs=None,
        self_instruction_mask=torch.ones(2, dtype=torch.float32),
        loss_agg_mode="token-mean",
        rollout_is_weights=None,
    )
    loss.backward()

    # Correct sample should increase log_prob (negative gradient on log_prob term).
    assert student_log_probs.grad[0, 0] < 0
    # Wrong sample should decrease log_prob (positive gradient on log_prob term).
    assert student_log_probs.grad[1, 0] > 0
    assert "sipo/pos_adv_mean" in metrics
    assert "sipo/neg_adv_mean" in metrics


def test_self_instruction_loss_uses_reward_mask_independent_of_distillation_mask():
    student_log_probs = torch.tensor([[-2.0], [-2.0]], dtype=torch.float32, requires_grad=True)
    teacher_log_probs = torch.tensor([[-2.0], [-2.0]], dtype=torch.float32)
    response_mask = torch.ones_like(student_log_probs)
    advantages = torch.tensor([[1.0], [-1.0]], dtype=torch.float32)
    cfg = SimpleNamespace(
        is_clip=None,
        alpha=1.0,
        full_logit_distillation=False,
    )

    loss, metrics = compute_self_instruction_loss(
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        response_mask=response_mask,
        self_distillation_config=cfg,
        advantages=advantages,
        old_log_probs=None,
        self_instruction_mask=torch.tensor([1.0, 0.0], dtype=torch.float32),
        loss_agg_mode="token-mean",
        rollout_is_weights=None,
    )
    loss.backward()

    # The second sample has no distillation target, but its GRPO reward gradient still applies.
    assert student_log_probs.grad[1, 0] > 0
    assert torch.isclose(torch.tensor(metrics["sipo/teacher_token_frac"]), torch.tensor(0.5))


def test_self_instruction_loss_uses_token_advantages():
    student_log_probs = torch.tensor([[-2.0]], dtype=torch.float32)
    teacher_log_probs = torch.tensor([[-2.0]], dtype=torch.float32)
    response_mask = torch.ones_like(student_log_probs)
    advantages = torch.tensor([[0.0]], dtype=torch.float32)
    cfg = SimpleNamespace(
        is_clip=None,
        alpha=1.0,
        full_logit_distillation=False,
    )

    loss, metrics = compute_self_instruction_loss(
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        response_mask=response_mask,
        self_distillation_config=cfg,
        advantages=advantages,
        old_log_probs=None,
        self_instruction_mask=torch.tensor([1.0], dtype=torch.float32),
        loss_agg_mode="token-mean",
        rollout_is_weights=None,
    )
    # With zero token-level advantages and zero divergence, total loss should be zero.
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(torch.tensor(metrics["sipo/adv_mean"]), torch.tensor(0.0), atol=1e-6)


def test_self_instruction_loss_beta_scales_divergence_term():
    student_log_probs = torch.tensor([[-0.5]], dtype=torch.float32)
    teacher_log_probs = torch.tensor([[-0.5]], dtype=torch.float32)
    response_mask = torch.ones_like(student_log_probs)
    advantages = torch.zeros_like(student_log_probs)
    student_all_log_probs = torch.log_softmax(torch.tensor([[[0.0, 0.0]]], dtype=torch.float32), dim=-1)
    teacher_all_log_probs = torch.log_softmax(torch.tensor([[[2.0, 0.0]]], dtype=torch.float32), dim=-1)
    base_cfg = SimpleNamespace(
        is_clip=None,
        alpha=0.0,
        beta=1.0,
        full_logit_distillation=True,
        distillation_topk=None,
    )
    scaled_cfg = SimpleNamespace(
        is_clip=None,
        alpha=0.0,
        beta=0.25,
        full_logit_distillation=True,
        distillation_topk=None,
    )

    base_loss, base_metrics = compute_self_instruction_loss(
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        response_mask=response_mask,
        self_distillation_config=base_cfg,
        advantages=advantages,
        old_log_probs=None,
        student_all_log_probs=student_all_log_probs,
        teacher_all_log_probs=teacher_all_log_probs,
        self_instruction_mask=torch.tensor([1.0], dtype=torch.float32),
        loss_agg_mode="token-mean",
        rollout_is_weights=None,
    )
    scaled_loss, scaled_metrics = compute_self_instruction_loss(
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        response_mask=response_mask,
        self_distillation_config=scaled_cfg,
        advantages=advantages,
        old_log_probs=None,
        student_all_log_probs=student_all_log_probs,
        teacher_all_log_probs=teacher_all_log_probs,
        self_instruction_mask=torch.tensor([1.0], dtype=torch.float32),
        loss_agg_mode="token-mean",
        rollout_is_weights=None,
    )

    assert torch.isclose(scaled_loss, base_loss * 0.25, atol=1e-6)
    assert scaled_metrics["sipo/divergence_loss"] == base_metrics["sipo/divergence_loss"]
    assert torch.isclose(
        torch.tensor(scaled_metrics["sipo/scaled_divergence_loss"]),
        torch.tensor(base_metrics["sipo/divergence_loss"] * 0.25),
        atol=1e-6,
    )
    assert scaled_metrics["sipo/divergence_beta"] == 0.25


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# RLSD: teacher-reweighted reward advantage (self_distillation.reinforce_advantage)
# ---------------------------------------------------------------------------


def _sipo_cfg(**overrides):
    cfg = dict(
        is_clip=None,
        alpha=1.0,
        beta=0.0,
        full_logit_distillation=False,
        reinforce_advantage=True,
        sipo_eps=0.2,
        sipo_lambda_start=0.5,
        sipo_lambda_steps=50,
    )
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def _pg_loss_for(advantages: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """token-mean PPO surrogate at ratio = 1 (old = current), which is -mean(A_t)."""
    return -(advantages * mask).sum() / mask.sum()


def _sipo_call(student, teacher, adv, cfg, mask=None, si_mask=None, global_step=None, debug=None, baseline_state=None, teacher_neg=None,
               all_wrong_row=None):
    mask = torch.ones_like(student) if mask is None else mask
    si_mask = torch.ones(student.shape[0], dtype=torch.float32) if si_mask is None else si_mask
    return compute_self_instruction_loss(
        student_log_probs=student,
        teacher_log_probs=teacher,
        response_mask=mask,
        self_distillation_config=cfg,
        advantages=adv,
        old_log_probs=student.detach().clone(),
        self_instruction_mask=si_mask,
        loss_agg_mode="token-mean",
        global_step=global_step,
        debug=debug,
        baseline_state=baseline_state,
        teacher_neg_log_probs=teacher_neg,
        all_wrong_row=all_wrong_row,
    )


def test_sipo_reweights_by_the_clipped_evidence_ratio():
    """A_t = A_r * ((1-lam) + lam * clip(exp(sign(A_r) * A_d), 1-eps, 1+eps)), no divergence term."""
    student = torch.tensor([[-2.0, -1.0, -0.5], [-1.0, -3.0, -0.1]], dtype=torch.float32, requires_grad=True)
    teacher = torch.tensor([[-0.5, -1.5, -0.5], [-0.2, -3.5, -0.1]], dtype=torch.float32)
    adv = torch.tensor([[0.5, 0.5, 0.5], [-0.875, -0.875, -0.875]], dtype=torch.float32)
    lam, eps = 0.5, 0.2
    loss, metrics = _sipo_call(student, teacher, adv, _sipo_cfg())
    a_d = teacher - student.detach()
    w = torch.exp(torch.sign(adv) * a_d).clamp(1 - eps, 1 + eps)
    expected_adv = adv * ((1 - lam) + lam * w)
    assert torch.isclose(loss, _pg_loss_for(expected_adv, torch.ones_like(adv)), atol=1e-6)
    assert metrics["sipo/divergence_loss"] == 0.0
    assert abs(metrics["sipo/lambda"] - lam) < 1e-9
    assert torch.all(torch.sign(expected_adv) == torch.sign(adv))
    ratio = expected_adv / adv
    assert torch.all((ratio >= 1 - lam * eps - 1e-6) & (ratio <= 1 + lam * eps + 1e-6))


def test_sipo_inverts_the_ratio_on_wrong_rollouts():
    """On A_r < 0 a token the teacher disfavours (A_d < 0) gets w > 1: pushed down harder."""
    student = torch.tensor([[-1.0, -1.0]], dtype=torch.float32, requires_grad=True)
    teacher = torch.tensor([[-3.0, -0.2]], dtype=torch.float32)   # token0 disfavoured, token1 favoured
    adv = torch.tensor([[-0.5, -0.5]], dtype=torch.float32)
    dbg = {}
    _sipo_call(student, teacher, adv, _sipo_cfg(sipo_lambda_start=1.0), debug=dbg)
    assert dbg["w"][0] > 1.0 and dbg["w"][1] < 1.0
    assert dbg["A_t"][0] < adv[0, 0] and dbg["A_t"][1] > adv[0, 1]   # more negative / less negative


def test_sipo_lambda_decays_to_plain_grpo():
    student = torch.tensor([[-2.0, -1.0]], dtype=torch.float32, requires_grad=True)
    teacher = torch.tensor([[-0.5, -9.0]], dtype=torch.float32)
    adv = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
    cfg = _sipo_cfg(sipo_lambda_start=0.5, sipo_lambda_steps=50)
    # Reference clock (iie-ycx/RLSD): lam = lam0 * (1 - step / steps), first update is step 1.
    _, m1 = _sipo_call(student, teacher, adv, cfg, global_step=1)
    _, m25 = _sipo_call(student, teacher, adv, cfg, global_step=25)
    loss_50, m50 = _sipo_call(student, teacher, adv, cfg, global_step=50)
    loss_200, m200 = _sipo_call(student, teacher, adv, cfg, global_step=200)
    assert abs(m1["sipo/lambda"] - 0.5 * (1 - 1 / 50)) < 1e-9
    assert abs(m25["sipo/lambda"] - 0.25) < 1e-9
    assert m50["sipo/lambda"] == 0.0 and m200["sipo/lambda"] == 0.0
    assert torch.isclose(loss_50, _pg_loss_for(adv, torch.ones_like(adv)), atol=1e-6)
    assert torch.isclose(loss_200, loss_50)


def test_sipo_leaves_zero_advantage_rows_and_unmasked_rows_alone():
    student = torch.tensor([[-2.0, -1.0], [-2.0, -1.0], [-2.0, -1.0]], dtype=torch.float32, requires_grad=True)
    teacher = torch.tensor([[-0.5, -9.0], [-0.5, -9.0], [-0.5, -9.0]], dtype=torch.float32)
    adv = torch.tensor([[0.0, 0.0], [0.5, 0.5], [0.5, 0.5]], dtype=torch.float32)
    si_mask = torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32)   # row 2 has no teacher context
    dbg = {}
    loss, _ = _sipo_call(student, teacher, adv, _sipo_cfg(sipo_lambda_start=1.0), si_mask=si_mask, debug=dbg)
    w = torch.exp(teacher[1] - student.detach()[1]).clamp(0.8, 1.2)
    expected = torch.stack([adv[0], adv[1] * w, adv[2]])
    assert torch.isclose(loss, _pg_loss_for(expected, torch.ones_like(adv)), atol=1e-6)
    assert dbg["row"] == 1   # the first row that is both masked and rewarded


def test_sipo_metrics_are_nan_when_no_row_is_active_and_reduce_ignores_them():
    import math

    from verl.utils.metric.utils import reduce_metrics

    student = torch.tensor([[-2.0, -1.0]], dtype=torch.float32, requires_grad=True)
    teacher = torch.tensor([[-0.5, -9.0]], dtype=torch.float32)
    _, idle = _sipo_call(student, teacher, torch.zeros(1, 2), _sipo_cfg())
    _, live = _sipo_call(student, teacher, torch.full((1, 2), 0.5), _sipo_cfg())
    assert set(idle) == set(live)
    assert math.isnan(idle["sipo/w_mean"]) and idle["sipo/active"] == 0.0 and live["sipo/active"] == 1.0
    reduced = reduce_metrics({"sipo/w_mean": [idle["sipo/w_mean"], live["sipo/w_mean"]]})
    assert abs(reduced["sipo/w_mean"] - live["sipo/w_mean"]) < 1e-9


def test_sipo_requires_reverse_kl_and_no_full_logits():
    student = torch.tensor([[-2.0, -1.0]], dtype=torch.float32, requires_grad=True)
    teacher = torch.tensor([[-0.5, -9.0]], dtype=torch.float32)
    adv = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
    with pytest.raises(ValueError, match="alpha=1.0"):
        _sipo_call(student, teacher, adv, _sipo_cfg(alpha=0.0))
    with pytest.raises(ValueError, match="full_logit_distillation=False"):
        _sipo_call(student, teacher, adv, _sipo_cfg(full_logit_distillation=True, distillation_topk=None))


def test_opd_stats_logged_even_when_flag_is_off():
    """The teacher's numbers are on the log without changing the objective."""
    student = torch.tensor([[-2.0, -1.0]], dtype=torch.float32, requires_grad=True)
    teacher = torch.tensor([[-0.5, -9.0]], dtype=torch.float32)
    _, metrics = _sipo_call(student, teacher, torch.tensor([[-0.25, -0.25]]), _sipo_cfg(reinforce_advantage=False, beta=1.0))
    assert metrics["sipo/reinforce_advantage"] == 0.0
    for key in ("opd/rms", "opd/p01", "opd/p99", "sipo/reward_adv_abs_mean", "opd/head_tail_gap"):
        assert key in metrics, key
    assert abs(metrics["sipo/reward_adv_abs_mean"] - 0.25) < 1e-6
    assert "sipo/lambda" not in metrics
    assert metrics["sipo/divergence_loss"] != 0.0


def test_format_sipo_debug_recomputes_the_formula():
    from verl.trainer.ppo.core_algos import format_sipo_debug

    student = torch.tensor([[-2.0, -1.0, -0.3, -0.01]], dtype=torch.float32, requires_grad=True)
    teacher = torch.tensor([[-0.5, -3.0, -0.3, -0.01]], dtype=torch.float32)
    adv = torch.full((1, 4), -0.875)
    dbg = {}
    _sipo_call(student, teacher, adv, _sipo_cfg(), debug=dbg)
    dbg["token_ids"] = torch.arange(4)
    out = format_sipo_debug(dbg, tokens=["Let", " me", "\n", " x"], step=3)
    assert "WRONG rollout" in out and "max err 0.0e+00" in out and "⏎" in out and "␣me" in out


# ---------------------------------------------------------------------------
# Group-advantage imputation (self_distillation.impute_group_advantage)
# ---------------------------------------------------------------------------


def _three_groups():
    """Groups of 4 rollouts: A all wrong, B mixed (2/4), C all correct. Dr.GRPO advantages."""
    from verl.trainer.ppo.core_algos import impute_group_advantage

    uids = ["A"] * 4 + ["B"] * 4 + ["C"] * 4
    scores = torch.tensor([0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1], dtype=torch.float32)
    T = 5
    mask = torch.ones(12, T); mask[1, 3:] = 0   # one short rollout in the all-wrong group
    adv = torch.zeros(12, T)
    adv[4:6] = 0.5; adv[6:8] = -0.5             # B: r - mean(r) with mean 0.5
    adv = adv * mask
    return impute_group_advantage, uids, scores, mask, adv


def test_impute_gives_all_wrong_groups_the_mixed_groups_wrong_advantage():
    fn, uids, scores, mask, adv = _three_groups()
    new_adv, m, imputed = fn(adv, mask, scores, uids, threshold=1.0)
    assert abs(m["impute/c"] - 0.5) < 1e-6                      # mean |A| of B's wrong rollouts
    assert torch.allclose(new_adv[0:4], -0.5 * mask[0:4])       # A imputed, padding respected
    assert torch.equal(new_adv[4:8], adv[4:8])                  # B untouched
    assert torch.equal(new_adv[8:12], torch.zeros(4, 5))        # C stays 0 by default
    assert imputed.tolist() == [True] * 4 + [False] * 8
    assert abs(m["all_wrong/row_frac"] - 4 / 12) < 1e-6 and abs(m["all_wrong/all_correct_row_frac"] - 4 / 12) < 1e-6


def test_impute_scales_and_optionally_rewards_all_correct_groups():
    fn, uids, scores, mask, adv = _three_groups()
    new_adv, _, imputed = fn(adv, mask, scores, uids, threshold=1.0, wrong_scale=0.5, correct_scale=0.25)
    assert torch.allclose(new_adv[0:4], -0.25 * mask[0:4])
    assert torch.allclose(new_adv[8:12], torch.full((4, 5), 0.125))
    assert imputed.tolist() == [True] * 4 + [False] * 4 + [True] * 4


def test_impute_is_a_no_op_without_a_mixed_group():
    fn, uids, scores, mask, adv = _three_groups()
    scores = scores.clone(); scores[4:8] = 0.0; adv = adv.clone(); adv[4:8] = 0.0   # B becomes all wrong
    import math
    new_adv, m, imputed = fn(adv, mask, scores, uids, threshold=1.0)
    assert torch.equal(new_adv, adv) and not imputed.any() and math.isnan(m["impute/c"])


def test_sipo_on_imputed_rows_only_leaves_mixed_groups_as_grpo():
    """The trainer restricts self_instruction_mask to imputed rows; the loss then reweights
    only those and returns the mixed groups' advantages bit-identical."""
    fn, uids, scores, mask, adv = _three_groups()
    new_adv, _, imputed = fn(adv, mask, scores, uids, threshold=1.0)
    torch.manual_seed(0)
    student = (-torch.rand(12, 5) * 2).requires_grad_(True)
    teacher = student.detach() + torch.randn(12, 5) * 0.5
    dbg = {}
    loss, metrics = compute_self_instruction_loss(
        student_log_probs=student, teacher_log_probs=teacher, response_mask=mask,
        self_distillation_config=_sipo_cfg(sipo_lambda_start=0.5, sipo_lambda_steps=0),
        advantages=new_adv, old_log_probs=student.detach().clone(),
        self_instruction_mask=imputed.float(), loss_agg_mode="token-mean", global_step=200, debug=dbg,
    )
    loss.backward()
    A_t = -student.grad * mask.sum()
    assert torch.allclose(A_t[4:12], new_adv[4:12], atol=1e-6)          # mixed + all-correct: exactly GRPO
    w = torch.exp(torch.sign(new_adv[0:4]) * (teacher - student.detach())[0:4]).clamp(0.8, 1.2)
    assert torch.allclose(A_t[0:4], new_adv[0:4] * (0.5 + 0.5 * w) * mask[0:4], atol=1e-6)
    assert metrics["sipo/lambda"] == 0.5 and dbg["row"] == 0 and dbg["A_r"] == -0.5


# ---------------------------------------------------------------------------
# RLSD evidence baseline (self_distillation.sipo_baseline=confidence) and negative-only
# ---------------------------------------------------------------------------


def _two_confidence_levels(seed=0):
    """Half the tokens are exploratory (logp_S = -5, bucket 0), half confident (logp_S = -0.01, bucket 6).
    The teacher's raw evidence is -0.8 on every exploratory token plus a token-specific part,
    and ~0 on confident ones: the 'teacher is more confident where you are unsure' profile."""
    torch.manual_seed(seed)
    B, T = 2, 400
    student = torch.where(torch.arange(T) % 2 == 0, torch.tensor(-5.0), torch.tensor(-0.01)).expand(B, T).clone()
    specific = torch.randn(B, T) * 0.3
    a_d = torch.where(student < -1, -0.8 + specific, 0.0 + 0.02 * specific)
    teacher = student + a_d
    return student.requires_grad_(True), teacher, a_d


def test_sipo_confidence_baseline_removes_the_confidence_trend():
    student, teacher, a_d = _two_confidence_levels()
    adv = torch.tensor([0.5, -0.5])[:, None].expand(2, 400).clone()
    dbg_raw, dbg_bl = {}, {}
    _sipo_call(student, teacher, adv, _sipo_cfg(sipo_lambda_start=1.0), debug=dbg_raw)
    _, m = _sipo_call(student, teacher, adv, _sipo_cfg(sipo_lambda_start=1.0, sipo_baseline="confidence"), debug=dbg_bl)
    explor = dbg_bl["logp_s"] < -1
    # Raw: every exploratory token of the correct rollout is de-weighted (w < 1, mean ~ exp(-0.8)).
    assert dbg_raw["w"][explor].mean() < 0.6
    # Centred: the baseline is ~ -0.8 on the exploratory bucket and ~ 0 on the confident one,
    # the centred evidence is zero-mean at both levels, and the weight no longer has a trend.
    assert abs(dbg_bl["base"][explor].mean() + 0.8) < 0.05 and abs(dbg_bl["base"][~explor].mean()) < 0.02
    assert abs(dbg_bl["A_hat"][explor].mean()) < 0.05 and abs(dbg_bl["A_hat"][~explor].mean()) < 0.02
    assert abs(dbg_bl["w"][explor].mean() - 1.0) < 0.06
    assert torch.allclose(dbg_bl["A_hat"], dbg_bl["A_d"] - dbg_bl["base"], atol=1e-6)
    assert abs(m["sipo/base/b0"] + 0.8) < 0.05 and abs(m["sipo/base/b6"]) < 0.02


def test_sipo_confidence_baseline_ema_blends_across_micro_batches():
    student, teacher, _ = _two_confidence_levels(seed=1)
    adv = torch.full((2, 400), 0.5)
    state = {}
    cfg = _sipo_cfg(sipo_baseline="confidence", sipo_baseline_ema=0.5)
    _, m1 = _sipo_call(student, teacher, adv, cfg, baseline_state=state)
    b_first = state["ema"].clone()
    teacher2 = teacher - 1.0                       # second micro batch: evidence shifted by -1 everywhere
    _, m2 = _sipo_call(student, teacher2, adv, cfg, baseline_state=state)
    assert abs(m2["sipo/base/b0"] - (0.5 * b_first[0].item() + 0.5 * (b_first[0].item() - 1.0))) < 0.05
    assert abs(m2["sipo/base/b6"] - (0.5 * b_first[6].item() + 0.5 * (b_first[6].item() - 1.0))) < 0.05


def test_sipo_negative_only_keeps_correct_rollouts_at_grpo():
    student = torch.tensor([[-2.0, -1.0], [-2.0, -1.0]], dtype=torch.float32, requires_grad=True)
    teacher = torch.tensor([[-0.5, -9.0], [-0.5, -9.0]], dtype=torch.float32)
    adv = torch.tensor([[0.5, 0.5], [-0.5, -0.5]], dtype=torch.float32)
    loss, _ = _sipo_call(student, teacher, adv, _sipo_cfg(sipo_lambda_start=1.0, sipo_negative_only=True))
    loss.backward()
    A_t = -student.grad * 4
    assert torch.allclose(A_t[0], adv[0], atol=1e-6)                           # correct row untouched
    w = torch.exp(-(teacher[1] - student.detach()[1])).clamp(0.8, 1.2)
    assert torch.allclose(A_t[1], adv[1] * w, atol=1e-6)                       # wrong row reweighted


# ---------------------------------------------------------------------------
# Contrastive evidence (self_distillation.sipo_evidence=contrastive)
# ---------------------------------------------------------------------------


def test_group_mode_wrong_answer_picks_the_groups_most_common_wrong_answer():
    from verl.trainer.ppo.core_algos import group_mode_wrong_answer

    uids = ["A"] * 4 + ["B"] * 3 + ["C"] * 2
    answers = ["5", "5", "7", "2", "2", "2", None, "9", "9"]
    ok = [False, False, False, True, True, True, False, False, False]
    out = group_mode_wrong_answer(answers, ok, uids)
    assert out[:4] == ["5"] * 4            # mixed group: mode of the WRONG answers, the correct "2" ignored
    assert out[4:7] == [None] * 3          # no parseable wrong answer (all correct except an unboxed one)
    assert out[7:] == ["9", "9"]           # all-wrong group still has a mode
    # ties resolve to the first seen
    assert group_mode_wrong_answer(["3", "4"], [False, False], ["u", "u"]) == ["3", "3"]
    # an answer equal to the gold string is never a candidate, even if the verifier failed the rollout
    assert group_mode_wrong_answer(["2", "2", "5"], [False, False, False], ["u"] * 3, golds=["2"] * 3) == ["5"] * 3
    assert group_mode_wrong_answer(["2", "2"], [False, False], ["u"] * 2, golds=["2"] * 2) == [None, None]


def test_own_or_group_wrong_answer_uses_the_rows_own_wrong_answer_and_falls_back_to_the_mode():
    from verl.trainer.ppo.core_algos import own_or_group_wrong_answer

    uids = ["A"] * 4 + ["B"] * 2
    answers = ["5", "7", None, "2", "9", "9"]
    ok = [False, False, False, True, False, False]
    out, own = own_or_group_wrong_answer(answers, ok, uids, golds=["2"] * 6)
    assert out == ["5", "7", None, "5", "9", "9"]     # wrong rows: own answer; truncated wrong row: none (skip); correct row: the mode "5"
    assert own == [True, True, False, False, True, True]
    out, _ = own_or_group_wrong_answer(answers, ok, uids, golds=["2"] * 6, unjudged="mode")
    assert out == ["5", "7", "5", "5", "9", "9"]      # the earlier fallback
    # a wrong rollout whose boxed answer equals the gold string never contrasts against gold
    out, own = own_or_group_wrong_answer(["2", "3"], [False, False], ["u", "u"], golds=["2", "2"])
    assert out == [None, "3"] and own == [False, True]       # the gold-equal wrong row has no answer of its own
    # no wrong answer anywhere: None, nothing used
    assert own_or_group_wrong_answer(["2"], [True], ["u"], golds=["2"]) == ([None], [False])


def test_contrastive_evidence_is_teacher_minus_wrong_teacher_and_ignores_student_confidence():
    """A token both teachers score identically gets w = 1 however unsure the student was;
    a token only the gold teacher likes gets w > 1 on a correct rollout."""
    student = torch.tensor([[-5.0, -5.0, -0.01, -0.01]], dtype=torch.float32, requires_grad=True)
    teacher = torch.tensor([[-1.0, -1.0, -0.5, -0.01]], dtype=torch.float32)          # gold-conditioned
    teacher_neg = torch.tensor([[-1.0, -2.0, -0.5, -0.31]], dtype=torch.float32)      # wrong-answer-conditioned
    adv = torch.full((1, 4), 0.5)
    dbg = {}
    loss, m = _sipo_call(student, teacher, adv, _sipo_cfg(sipo_lambda_start=1.0, sipo_evidence="contrastive"),
                         debug=dbg, teacher_neg=teacher_neg)
    e = teacher - teacher_neg
    assert torch.allclose(dbg["A_d"], e[0], atol=1e-6)                    # evidence is T - T-, not T - S
    assert abs(dbg["w"][0] - 1.0) < 1e-6 and abs(dbg["w"][2] - 1.0) < 1e-6   # equal under both contexts -> untouched
    assert abs(dbg["w"][1] - math.e ** 1.0) < 1e-5                        # gold teacher likes it more -> w = exp(+1)
    assert abs(dbg["w"][3] - math.e ** 0.3) < 1e-5
    loss.backward()
    A_t = -student.grad * 4
    assert torch.allclose(A_t[0], adv[0] * e.exp().clamp(0.8, 1.2)[0], atol=1e-6)
    assert "contrast/rms" in m and "opd/rms" in m                  # both evidences logged


def test_contrastive_evidence_requires_the_second_teacher():
    student = torch.tensor([[-2.0, -1.0]], dtype=torch.float32, requires_grad=True)
    teacher = torch.tensor([[-0.5, -9.0]], dtype=torch.float32)
    with pytest.raises(ValueError, match="teacher_neg_log_probs"):
        _sipo_call(student, teacher, torch.full((1, 2), 0.5), _sipo_cfg(sipo_evidence="contrastive"))


def test_contrastive_metric_keys_match_between_idle_and_live_micro_batches():
    student = torch.tensor([[-2.0, -1.0]], dtype=torch.float32, requires_grad=True)
    teacher = torch.tensor([[-0.5, -9.0]], dtype=torch.float32); teacher_neg = teacher - 0.3
    cfg = _sipo_cfg(sipo_evidence="contrastive")
    _, idle = _sipo_call(student, teacher, torch.zeros(1, 2), cfg, teacher_neg=teacher_neg)
    _, live = _sipo_call(student, teacher, torch.full((1, 2), 0.5), cfg, teacher_neg=teacher_neg)
    assert set(idle) == set(live)


def test_loss_metric_names_follow_the_section_scheme():
    """Every loss metric sits in one of the README's wandb sections, and only the intended keys
    contain "min"/"max", which reduce_metrics reads as the reduction to apply."""
    student = torch.tensor([[-2.0, -1.0], [-1.0, -0.5]], dtype=torch.float32, requires_grad=True)
    teacher = torch.tensor([[-0.5, -9.0], [-0.7, -0.4]], dtype=torch.float32); teacher_neg = teacher - 0.3
    adv = torch.tensor([[0.5, 0.5], [0.0, 0.0]])
    for cfg in (_sipo_cfg(sipo_evidence="contrastive", sipo_form="additive", sipo_kappa=0.15),
                _sipo_cfg(sipo_evidence="contrastive", sipo_all_wrong_kappa=0.15, sipo_baseline="confidence"),
                _sipo_cfg(reinforce_advantage=False, beta=1.0)):
        _, m = _sipo_call(student, teacher, adv, cfg, teacher_neg=teacher_neg, all_wrong_row=torch.tensor([0.0, 1.0]))
        assert {k.split("/")[0] for k in m} <= {"sipo", "opd", "contrast", "sipo", "all_wrong"}, sorted(m)
        assert {k for k in m if "min" in k or "max" in k} == {"opd/min", "opd/max"}, sorted(m)


def test_format_sipo_debug_shows_the_wrong_teacher_column():
    from verl.trainer.ppo.core_algos import format_sipo_debug

    student = torch.tensor([[-2.0, -1.0, -0.3]], dtype=torch.float32, requires_grad=True)
    teacher = torch.tensor([[-0.5, -3.0, -0.3]], dtype=torch.float32); teacher_neg = teacher + torch.tensor([[0.2, -0.4, 0.0]])
    dbg = {}
    _sipo_call(student, teacher, torch.full((1, 3), -0.5), _sipo_cfg(sipo_evidence="contrastive"), debug=dbg, teacher_neg=teacher_neg)
    dbg["token_ids"] = torch.arange(3)
    out = format_sipo_debug(dbg, tokens=["a", "b", "c"], step=2)
    assert "evidence=contrastive" in out and "logp_T-" in out and "A_d=logp_T-logp_T-: max err 0.0e+00" in out


# Additive contrastive credit on all-wrong groups (self_distillation.sipo_all_wrong_kappa)


def test_all_wrong_group_rows_marks_zero_advantage_rows_of_all_wrong_groups_and_logs_c():
    from verl.trainer.ppo.core_algos import all_wrong_group_rows

    # g1 mixed (1 right, 1 wrong), g2 all wrong (A_r = 0), g3 all correct
    adv = torch.tensor([[0.5, 0.5], [-0.5, -0.5], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    scores = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0])
    uids = ["g1", "g1", "g2", "g2", "g3"]
    rows, m = all_wrong_group_rows(adv, torch.ones_like(adv), scores, uids, threshold=1.0)
    assert torch.equal(rows, torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0]))
    assert abs(m["all_wrong/c"] - 0.5) < 1e-6                      # the mixed group's wrong row has |A_r| = 0.5
    assert abs(m["all_wrong/row_frac"] - 0.4) < 1e-6 and abs(m["all_wrong/marked_row_frac"] - 0.4) < 1e-6
    assert abs(m["all_wrong/all_correct_row_frac"] - 0.2) < 1e-6


def test_all_wrong_group_rows_skips_all_wrong_rows_whose_advantage_is_not_zero():
    from verl.trainer.ppo.core_algos import all_wrong_group_rows

    # g2 is all wrong by the threshold, but a shaped reward (0 vs -1) gives it a nonzero A_r:
    # those rows already go through the multiplicative path and must not be counted twice.
    adv = torch.tensor([[0.5, 0.5], [-0.5, -0.5], [0.5, 0.5], [-0.5, -0.5]])
    scores = torch.tensor([1.0, 0.0, 0.0, -1.0]); uids = ["g1", "g1", "g2", "g2"]
    rows, m = all_wrong_group_rows(adv, torch.ones_like(adv), scores, uids, threshold=1.0)
    assert torch.all(rows == 0) and abs(m["all_wrong/row_frac"] - 0.5) < 1e-6 and m["all_wrong/marked_row_frac"] == 0.0


def test_all_wrong_group_rows_does_not_need_a_mixed_group():
    from verl.trainer.ppo.core_algos import all_wrong_group_rows

    adv = torch.zeros(2, 2); scores = torch.tensor([0.0, 0.0]); uids = ["g", "g"]
    rows, m = all_wrong_group_rows(adv, torch.ones_like(adv), scores, uids, threshold=1.0)
    assert torch.equal(rows, torch.tensor([1.0, 1.0])) and math.isnan(m["all_wrong/c"])   # kappa is a constant; c is only logged


def test_sipo_all_wrong_rows_get_kappa_times_the_clipped_contrast_and_mixed_rows_are_untouched():
    """Row 0 sits in a mixed group (A_r = 0.5): the multiplicative form, unchanged by the flag.
    Row 1 sits in an all-wrong group (A_r = 0): A_t = kappa * clip(e_t, +-eps), kappa = sipo_all_wrong_kappa."""
    student = torch.tensor([[-2.0, -1.0, -0.5], [-1.0, -3.0, -0.1]], dtype=torch.float32, requires_grad=True)
    teacher = torch.tensor([[-0.5, -1.5, -0.5], [-0.2, -3.5, -0.1]], dtype=torch.float32)
    teacher_neg = teacher - torch.tensor([[0.1, -0.5, 0.0], [0.05, 0.0, -0.3]])   # e = T - T-: row 1 is +0.05, 0, -0.3
    adv = torch.tensor([[0.5, 0.5, 0.5], [0.0, 0.0, 0.0]], dtype=torch.float32)
    rows = torch.tensor([0.0, 1.0])
    cfg = _sipo_cfg(sipo_evidence="contrastive", sipo_all_wrong_kappa=0.15, sipo_lambda_start=0.5, sipo_lambda_steps=0)
    dbg = {}
    loss, m = _sipo_call(student, teacher, adv, cfg, teacher_neg=teacher_neg, all_wrong_row=rows, debug=dbg)
    loss.backward()
    A_t = -student.grad * 6
    e = teacher - teacher_neg
    lam, eps, kappa = 0.5, 0.2, 0.15
    expected_row0 = adv[0] * ((1 - lam) + lam * torch.exp(e[0]).clamp(1 - eps, 1 + eps))
    expected_row1 = kappa * e[1].clamp(-eps, eps)                                   # 0.15 * [0.05, 0, -0.2]
    assert torch.allclose(A_t[0], expected_row0, atol=1e-6)
    assert torch.allclose(A_t[1], expected_row1, atol=1e-6)
    assert m["all_wrong/active"] == 1.0 and abs(m["all_wrong/kappa"] - kappa) < 1e-6
    assert abs(m["all_wrong/at_clip_frac"] - 1 / 3) < 1e-6 and abs(m["all_wrong/nonzero_frac"] - 2 / 3) < 1e-6
    assert abs(m["all_wrong/row_sum_mean"] - float(expected_row1.sum())) < 1e-6
    assert dbg["aw_row"] == 1 and torch.allclose(dbg["aw_A"], expected_row1, atol=1e-6) and abs(dbg["aw_kappa"] - kappa) < 1e-6
    assert dbg["row"] == 0                                                          # the mixed row is still rendered too
    # Flag at 0: both rows as before, same metric keys (NaN placeholders), row 1 gets nothing.
    student2 = student.detach().clone().requires_grad_(True)
    loss_off, m_off = _sipo_call(student2, teacher, adv, _sipo_cfg(sipo_evidence="contrastive", sipo_lambda_start=0.5, sipo_lambda_steps=0),
                                 teacher_neg=teacher_neg)
    loss_off.backward()
    A_off = -student2.grad * 6
    assert torch.allclose(A_off[0], expected_row0, atol=1e-6) and torch.all(A_off[1] == 0)
    assert m_off["all_wrong/active"] == 0.0 and math.isnan(m_off["all_wrong/kappa"]) and set(m_off) == set(m)
    # Flag on but no row tensor shipped (e.g. the trainer left it out): nothing is added.
    student3 = student.detach().clone().requires_grad_(True)
    loss_noscale, _ = _sipo_call(student3, teacher, adv, cfg, teacher_neg=teacher_neg)
    loss_noscale.backward()
    assert torch.all((-student3.grad * 6)[1] == 0)


def test_sipo_all_wrong_credit_requires_contrastive_evidence():
    student = torch.tensor([[-2.0, -1.0]], dtype=torch.float32, requires_grad=True)
    teacher = torch.tensor([[-0.5, -9.0]], dtype=torch.float32)
    with pytest.raises(ValueError, match="contrastive"):
        _sipo_call(student, teacher, torch.zeros(1, 2), _sipo_cfg(sipo_all_wrong_kappa=1.0), all_wrong_row=torch.tensor([1.0]))


def test_format_sipo_debug_renders_the_all_wrong_block():
    from verl.trainer.ppo.core_algos import format_sipo_debug

    student = torch.tensor([[-2.0, -1.0, -0.3]], dtype=torch.float32, requires_grad=True)
    teacher = torch.tensor([[-0.5, -3.0, -0.3]], dtype=torch.float32); teacher_neg = teacher + torch.tensor([[0.2, -0.4, 0.0]])
    dbg = {}
    _sipo_call(student, teacher, torch.zeros(1, 3), _sipo_cfg(sipo_evidence="contrastive", sipo_all_wrong_kappa=0.2, sipo_lambda_steps=0),
               teacher_neg=teacher_neg, all_wrong_row=torch.tensor([1.0]), debug=dbg)
    assert "row" not in dbg and dbg["aw_row"] == 0          # no mixed row in this micro batch: only the all-wrong block
    dbg["aw_token_ids"] = torch.arange(3)
    out = format_sipo_debug(dbg, step=3, aw_tokens=["a", "b", "c"])
    assert "ALL-WRONG group" in out and "A_t=kappa*clip(e): max err 0.0e+00" in out and "kappa=0.2000" in out
    assert "CREDIT UP" in out and "sum A_t" in out


def test_all_wrong_kappa_config_validation():
    cfgmod = pytest.importorskip("verl.workers.config.actor")
    C = cfgmod.SelfDistillationConfig
    base = dict(full_logit_distillation=False, alpha=1.0, reinforce_advantage=True, sipo_evidence="contrastive")
    C(**base, sipo_all_wrong_kappa=1.0)                                         # valid
    C(**{**base, "sipo_evidence": "opd"}, sipo_all_wrong_kappa=0.0)              # off: evidence unconstrained
    with pytest.raises(ValueError, match="contrastive"):
        C(**{**base, "sipo_evidence": "opd"}, sipo_all_wrong_kappa=1.0)
    with pytest.raises(ValueError, match="reinforce_advantage"):
        C(**{**base, "reinforce_advantage": False}, sipo_all_wrong_kappa=1.0)
    with pytest.raises(ValueError, match="impute_group_advantage"):
        C(**base, sipo_all_wrong_kappa=1.0, impute_group_advantage=True)
    with pytest.raises(ValueError, match=">= 0"):
        C(**base, sipo_all_wrong_kappa=-0.5)


def test_judged_answers_uses_the_verifiers_pred_and_counts_unjudged_boxes():
    from verl.trainer.ppo.core_algos import judged_answers

    texts = ["so \\boxed{40^\\circ}. Let me double check this at length ... (cut off)", "thus \\boxed{7}", "no box at all"]
    ans, n, _ = judged_answers(["", "7", ""], texts)    # the verifier saw no box in rows 0 and 2
    assert ans == [None, "7", None] and n == 1          # row 0 has an early box the verifier never judged
    ans, n, _ = judged_answers(None, texts)             # reward manager reported no preds: whole-response fallback
    assert ans == ["40^\\circ", "7", None] and n == 0
    assert judged_answers([None, " 7 "], texts[:2])[:2] == ([None, "7"], 1)   # None counts as unjudged too; row 0 still has its early box


# Additive form (self_distillation.sipo_form=additive)


def test_sipo_additive_form_adds_kappa_clip_e_to_every_row_and_keeps_signs_at_small_kappa():
    """A_t = A_r + kappa*clip(e_t, +-eps) on every row: a 7-of-8 group's correct row (|A_r| = 1/8,
    the smallest a mixed row can have), a wrong row, and an all-wrong row (A_r = 0)."""
    student = torch.tensor([[-2.0, -1.0, -0.5], [-1.0, -3.0, -0.1], [-1.5, -0.2, -0.3]], dtype=torch.float32, requires_grad=True)
    teacher = torch.tensor([[-0.5, -1.5, -0.5], [-0.2, -3.5, -0.1], [-0.4, -0.2, -0.9]], dtype=torch.float32)
    e = torch.tensor([[0.1, -0.5, 0.0], [0.05, 0.0, -0.3], [-0.6, 0.02, 0.3]])
    teacher_neg = teacher - e
    adv = torch.tensor([[0.125] * 3, [-0.875] * 3, [0.0] * 3])
    cfg = _sipo_cfg(sipo_form="additive", sipo_kappa=0.15, sipo_evidence="contrastive")
    dbg = {}
    loss, m = _sipo_call(student, teacher, adv, cfg, teacher_neg=teacher_neg, all_wrong_row=torch.tensor([0.0, 0.0, 1.0]), debug=dbg)
    loss.backward()
    A_t = -student.grad * 9
    expected = adv + 0.15 * e.clamp(-0.2, 0.2)
    assert torch.allclose(A_t, expected, atol=1e-6)
    assert torch.all(torch.sign(A_t[:2]) == torch.sign(adv[:2]))       # clamp never engaged: kappa*eps = 0.03 < 1/8
    assert m["sipo/form_additive"] == 1.0 and abs(m["sipo/kappa"] - 0.15) < 1e-9 and math.isnan(m["sipo/lambda"])
    assert m["sipo/clamp_frac"] == 0.0 and math.isnan(m["sipo/w_mean"])
    assert m["all_wrong/active"] == 1.0 and abs(m["all_wrong/kappa"] - 0.15) < 1e-9
    assert torch.allclose(dbg["aw_A"], expected[2], atol=1e-6) and dbg["aw_row"] == 2
    assert dbg["row"] == 0 and dbg["form"] == "additive"
    # the all-wrong row is found from the advantages when the trainer did not ship the mark
    student2 = student.detach().clone().requires_grad_(True)
    _, m2 = _sipo_call(student2, teacher, adv, cfg, teacher_neg=teacher_neg)
    assert m2["all_wrong/active"] == 1.0 and abs(m2["all_wrong/dA_rms"] - m["all_wrong/dA_rms"]) < 1e-9


def test_sipo_additive_form_sign_clamp_engages_only_when_kappa_eps_exceeds_the_advantage():
    student = torch.tensor([[-1.0, -1.0]], dtype=torch.float32, requires_grad=True)
    teacher = torch.tensor([[-0.5, -0.5]], dtype=torch.float32)
    teacher_neg = teacher - torch.tensor([[-0.5, 0.5]])             # e = -0.5, +0.5 -> clip -0.2, +0.2
    adv = torch.full((1, 2), 0.05)                                  # smaller than kappa*eps = 0.1
    cfg = _sipo_cfg(sipo_form="additive", sipo_kappa=0.5, sipo_evidence="contrastive")
    loss, m = _sipo_call(student, teacher, adv, cfg, teacher_neg=teacher_neg)
    loss.backward()
    A_t = -student.grad * 2
    assert torch.allclose(A_t, torch.tensor([[0.0, 0.15]]), atol=1e-6)   # 0.05 - 0.1 -> clamped to 0; 0.05 + 0.1
    assert abs(m["sipo/clamp_frac"] - 0.5) < 1e-9


def test_sipo_additive_form_negative_only_leaves_correct_rows_at_grpo_and_keeps_all_wrong_rows():
    student = torch.full((3, 2), -1.0).requires_grad_(True)
    teacher = torch.full((3, 2), -0.5); teacher_neg = teacher - 0.1      # e = +0.1 everywhere
    adv = torch.tensor([[0.5, 0.5], [-0.5, -0.5], [0.0, 0.0]])
    cfg = _sipo_cfg(sipo_form="additive", sipo_kappa=0.15, sipo_evidence="contrastive", sipo_negative_only=True)
    loss, _ = _sipo_call(student, teacher, adv, cfg, teacher_neg=teacher_neg)
    loss.backward()
    A_t = -student.grad * 6
    assert torch.allclose(A_t, torch.tensor([[0.5, 0.5], [-0.485, -0.485], [0.015, 0.015]]), atol=1e-6)


def test_sipo_additive_form_requires_contrastive_evidence_and_a_positive_kappa():
    student = torch.tensor([[-2.0, -1.0]], dtype=torch.float32, requires_grad=True)
    teacher = torch.tensor([[-0.5, -9.0]], dtype=torch.float32)
    with pytest.raises(ValueError, match="contrastive"):
        _sipo_call(student, teacher, torch.full((1, 2), 0.5), _sipo_cfg(sipo_form="additive", sipo_kappa=0.15))
    with pytest.raises(ValueError, match="sipo_kappa"):
        _sipo_call(student, teacher, torch.full((1, 2), 0.5), _sipo_cfg(sipo_form="additive", sipo_kappa=0.0, sipo_evidence="contrastive"),
                   teacher_neg=teacher - 0.1)


def test_format_sipo_debug_recomputes_the_additive_formula():
    from verl.trainer.ppo.core_algos import format_sipo_debug

    student = torch.tensor([[-2.0, -1.0, -0.3]], dtype=torch.float32, requires_grad=True)
    teacher = torch.tensor([[-0.5, -3.0, -0.3]], dtype=torch.float32); teacher_neg = teacher + torch.tensor([[0.2, -0.4, 0.0]])
    dbg = {}
    _sipo_call(student, teacher, torch.full((1, 3), -0.5), _sipo_cfg(sipo_form="additive", sipo_kappa=0.15, sipo_evidence="contrastive"),
               teacher_neg=teacher_neg, debug=dbg)
    dbg["token_ids"] = torch.arange(3)
    out = format_sipo_debug(dbg, tokens=["a", "b", "c"], step=2)
    assert "kappa=0.1500" in out and "A_t=clamp_sign(A_r+kappa*clip(A^)): max err 0.0e+00" in out and "clipA^" in out
    assert "aw_row" not in dbg                                       # a single mixed row: no all-wrong block


def test_additive_form_config_validation():
    cfgmod = pytest.importorskip("verl.workers.config.actor")
    C = cfgmod.SelfDistillationConfig
    base = dict(full_logit_distillation=False, alpha=1.0, reinforce_advantage=True, sipo_evidence="contrastive")
    C(**base, sipo_form="additive", sipo_kappa=0.15)
    with pytest.raises(ValueError, match="sipo_kappa"):
        C(**base, sipo_form="additive")
    with pytest.raises(ValueError, match="contrastive"):
        C(**{**base, "sipo_evidence": "opd"}, sipo_form="additive", sipo_kappa=0.15)
    with pytest.raises(ValueError, match="sipo_all_wrong_kappa"):
        C(**base, sipo_form="additive", sipo_kappa=0.15, sipo_all_wrong_kappa=0.15)
    with pytest.raises(ValueError, match="sipo_form"):
        C(**base, sipo_form="linear")


def test_tooluse_reference_answer_matches_the_scorers_prediction_format():
    from verl.trainer.ppo.core_algos import tooluse_reference_answer
    import json

    gt = json.dumps([{"Action": "search", "Action_Input": json.dumps({"query": "x"})},
                     {"Action": "calc", "Action_Input": {"expr": "1+1"}},
                     {"Action": "noop", "Action_Input": "not json"}])
    # feedback/tooluse.py renders a prediction as f"Actions: {actions}, Inputs: {merged inputs}"
    assert tooluse_reference_answer(gt) == "Actions: ['search', 'calc', 'noop'], Inputs: {'query': 'x', 'expr': '1+1'}"
    assert tooluse_reference_answer("not json") is None and tooluse_reference_answer(None) is None
    assert tooluse_reference_answer([{"Action": "a", "Action_Input": {}}]) == "Actions: ['a'], Inputs: {}"


def test_render_reference_answer_dispatches_by_task():
    from verl.trainer.ppo.core_algos import render_reference_answer

    assert render_reference_answer("math_dapo", " 40 ") == "40"
    assert render_reference_answer("gpqa", "C") == "C" and render_reference_answer("sciknoweval", "B") == "B"
    assert render_reference_answer("tooluse", '[{"Action": "a", "Action_Input": {}}]') == "Actions: ['a'], Inputs: {}"
    assert render_reference_answer("livecodebench", '{"tests": []}') is None and render_reference_answer("code", "x") is None
    assert render_reference_answer(None, "") is None and render_reference_answer("math", None) is None


def test_contrast_context_config_validation():
    cfgmod = pytest.importorskip("verl.workers.config.actor")
    C = cfgmod.SelfDistillationConfig
    base = dict(full_logit_distillation=False, alpha=1.0, reinforce_advantage=True)
    C(**base, sipo_evidence="contrastive", sipo_contrast_context="trajectory")
    C(**base, sipo_evidence="contrastive", sipo_contrast_context="auto")
    with pytest.raises(ValueError, match="contrastive"):
        C(**base, sipo_evidence="opd", sipo_contrast_context="trajectory")
    with pytest.raises(ValueError, match="contrastive"):
        C(**base, sipo_evidence="opd", sipo_contrast_context="auto")
    with pytest.raises(ValueError, match="sipo_contrast_context"):
        C(**base, sipo_evidence="contrastive", sipo_contrast_context="answers")


def test_valid_judged_answer_rejects_preds_that_are_not_answers():
    from verl.trainer.ppo.core_algos import valid_judged_answer

    essay = "<reasoning>The octanol/water partition ... (cut off, no answer tag)"
    assert valid_judged_answer("sciknoweval", "C") == "C" and valid_judged_answer("gpqa", " B ") == "B"
    assert valid_judged_answer("sciknoweval", essay) is None                 # mcq.py returns the whole text
    assert valid_judged_answer("sciknoweval", "C, because the logD is") is None
    assert valid_judged_answer("gpqa", "the answer is probably") is None      # gpqa.py falls back to the text
    assert valid_judged_answer("tooluse", "Actions: ['a'], Inputs: {}") == "Actions: ['a'], Inputs: {}"
    assert valid_judged_answer("tooluse", "no actions here") is None
    assert valid_judged_answer("livecodebench", "anything") is None
    assert valid_judged_answer("math_dapo", "40^\\circ") == "40^\\circ" and valid_judged_answer("math_dapo", "  ") is None


def test_judged_answers_validates_per_task_and_counts_rejections():
    from verl.trainer.ppo.core_algos import judged_answers

    texts = ["<answer>C</answer>", "rambling, cut off", "so \\boxed{8}"]
    ans, n_unj, n_inv = judged_answers(["C", "rambling, cut off", "8"], texts, data_sources=["sciknoweval", "sciknoweval", "math_dapo"])
    assert ans == ["C", None, "8"] and n_inv == 1 and n_unj == 0


def test_own_or_group_wrong_answer_skips_wrong_rows_without_their_own_answer():
    from verl.trainer.ppo.core_algos import own_or_group_wrong_answer

    answers = ["8", None, "8", "6"]                  # row 1 is a truncated wrong rollout, row 3 correct
    ok = [False, False, False, True]
    uids = ["g"] * 4
    out, own = own_or_group_wrong_answer(answers, ok, uids, golds=["6"] * 4)          # default skip
    assert out == ["8", None, "8", "8"] and own == [True, False, True, False]
    out, own = own_or_group_wrong_answer(answers, ok, uids, golds=["6"] * 4, unjudged="mode")
    assert out == ["8", "8", "8", "8"] and own == [True, False, True, False]
    with pytest.raises(ValueError, match="unjudged"):
        own_or_group_wrong_answer(answers, ok, uids, unjudged="drop")


def test_extract_boxed_answer_matches_the_reward_extractor_rule():
    from verl.trainer.ppo.core_algos import extract_boxed_answer

    assert extract_boxed_answer("so the tens digit is\n$$\n\\boxed{2}\n$$<|im_end|>") == "2"
    assert extract_boxed_answer("first \\boxed{5} then final \\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"
    assert extract_boxed_answer("nested \\boxed{\\sqrt{x^{2}}}") == "\\sqrt{x^{2}}"
    assert extract_boxed_answer("no box at all") is None
    assert extract_boxed_answer("unbalanced \\boxed{3") is None
    assert extract_boxed_answer("empty \\boxed{}") is None
    assert extract_boxed_answer(None) is None
