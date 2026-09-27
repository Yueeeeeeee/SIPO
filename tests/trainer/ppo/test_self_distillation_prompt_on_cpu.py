import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


STUDENT_SUFFIX = "\n\nPlease reason step by step before generating your final answer."


class _FakeTokenizer:
    def __init__(self):
        self.messages = None
        self._decode_map = {
            10: "correct target A",
            11: "wrong peer A without feedback",
            12: "correct target B1",
            13: "correct target B2",
            14: "wrong peer A with feedback",
            15: "correct target C",
            16: "wrong peer C1",
            17: "wrong peer C2",
            18: "wrong peer C3",
        }

    def decode(self, ids, skip_special_tokens=True):
        token_ids = ids.detach().cpu().tolist() if isinstance(ids, torch.Tensor) else list(ids)
        return " ".join(self._decode_map.get(token_id, f"<{token_id}>") for token_id in token_ids if token_id != 0)

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        batch_size = len(messages)
        return {
            "input_ids": torch.ones(batch_size, 3, dtype=torch.long),
            "attention_mask": torch.ones(batch_size, 3, dtype=torch.long),
        }


def _make_trainer():
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.tokenizer = _FakeTokenizer()
    trainer.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "actor": {
                    "policy_loss": {"loss_mode": "sipo"},
                    "self_distillation": {
                        "include_environment_feedback": True,
                        "environment_feedback_only_without_solution": True,
                        "success_reward_threshold": 0.5,
                        "dont_reprompt_on_self_success": True,
                        "remove_thinking_from_demonstration": False,
                        "student_reasoning_suffix": STUDENT_SUFFIX,
                        "max_reprompt_len": 128,
                        "reprompt_template": (
                            "{prompt}{solution}{incorrect_attempt}{feedback}"
                            "\n\nUse the context above to solve the original question correctly."
                        ),
                        "solution_template": "\n\n# Successful reference solution:\n\n{successful_previous_attempt}",
                        "incorrect_attempt_template": (
                            "\n\n# Incorrect reference attempt:\n\n{incorrect_attempt}"
                        ),
                        "feedback_template": (
                            "\n\n# Environment feedback for the incorrect reference attempt:\n\n{feedback_raw}"
                        ),
                    },
                }
            },
            "data": {"apply_chat_template_kwargs": {"enable_thinking": True}},
        }
    )
    return trainer


def test_correct_rollout_prefers_incorrect_peer_context_with_feedback():
    trainer = _make_trainer()
    batch = DataProto.from_dict(
        tensors={
            "input_ids": torch.ones(9, 4, dtype=torch.long),
            "attention_mask": torch.ones(9, 4, dtype=torch.long),
            "responses": torch.tensor(
                [[10, 0], [11, 0], [14, 0], [12, 0], [13, 0], [15, 0], [16, 0], [17, 0], [18, 0]],
                dtype=torch.long,
            ),
            "response_mask": torch.ones(9, 2, dtype=torch.long),
        },
        non_tensors={
            "uid": [
                "group-a",
                "group-a",
                "group-a",
                "group-b",
                "group-b",
                "group-c",
                "group-c",
                "group-c",
                "group-c",
            ],
            "raw_prompt": [
                [{"role": "user", "content": f"Question A{STUDENT_SUFFIX}"}],
                [{"role": "user", "content": f"Question A{STUDENT_SUFFIX}"}],
                [{"role": "user", "content": f"Question A{STUDENT_SUFFIX}"}],
                [{"role": "user", "content": f"Question B{STUDENT_SUFFIX}"}],
                [{"role": "user", "content": f"Question B{STUDENT_SUFFIX}"}],
                [{"role": "user", "content": f"Question C{STUDENT_SUFFIX}"}],
                [{"role": "user", "content": f"Question C{STUDENT_SUFFIX}"}],
                [{"role": "user", "content": f"Question C{STUDENT_SUFFIX}"}],
                [{"role": "user", "content": f"Question C{STUDENT_SUFFIX}"}],
            ],
        },
    )
    reward_tensor = torch.tensor([[1.0], [0.0], [0.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0]])
    reward_extra_infos = {"feedback": ["", "", "preferred wrong peer feedback", "", "", "", "", "", ""]}

    self_distillation_batch, metrics = RayPPOTrainer._maybe_build_self_distillation_batch(
        trainer,
        batch,
        reward_tensor,
        reward_extra_infos,
    )

    prompt_for_correct_with_error = trainer.tokenizer.messages[0][-1]["content"]
    assert "# Incorrect reference attempt:" in prompt_for_correct_with_error
    assert "wrong peer A with feedback" in prompt_for_correct_with_error
    assert "wrong peer A without feedback" not in prompt_for_correct_with_error
    assert "# Environment feedback for the incorrect reference attempt:" in prompt_for_correct_with_error
    assert "preferred wrong peer feedback" in prompt_for_correct_with_error
    assert "# Successful reference solution:" not in prompt_for_correct_with_error

    prompt_for_correct_without_error = trainer.tokenizer.messages[3][-1]["content"]
    assert "# Successful reference solution:" in prompt_for_correct_without_error
    assert "correct target B2" in prompt_for_correct_without_error
    assert "# Incorrect reference attempt:" not in prompt_for_correct_without_error

    prompt_for_correct_with_errors_without_feedback = trainer.tokenizer.messages[5][-1]["content"]
    assert "wrong peer C1" in prompt_for_correct_with_errors_without_feedback
    assert "wrong peer C2" not in prompt_for_correct_with_errors_without_feedback

    assert metrics["teacher/ctx_incorrect_ref_frac"] == 2 / 9
    assert metrics["self_distillation/feedback_used_fraction"] == 1 / 9
    assert {k.split("/")[0] for k in metrics} <= {"self_distillation", "teacher", "contrast", "impute"}
    assert torch.all(self_distillation_batch.batch["self_instruction_mask"] == 1)
