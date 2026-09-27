#!/usr/bin/env python
"""Tokenize the actor and teacher prompts the way training does, and diff them.

    python scripts/check_teacher_prompt.py                 # repo defaults
    python scripts/check_teacher_prompt.py --sipo-ctx      # what SIPO_CTX=1 sends

The divergence A_d = log pi_teacher(y|x_aug) - log pi_student(y|x) is only a measure of
hindsight if x_aug and x differ by the privileged information and nothing else. Anything
else that differs -- a changed closing instruction, a dropped system turn, a different
chat-template kwarg -- shows up in A_d as signal. This reproduces both sides end to end
against the real tokenizer and reports exactly what separates them.

Student:  agent_loop.py:297 tokenizes raw_prompt with add_generation_prompt=True,
          tokenize=True and data.apply_chat_template_kwargs.
Teacher:  ray_trainer.py:860 tokenizes raw_prompt[:-1] + [user=reprompt] with the same
          add_generation_prompt and enable_thinking.
"""
import argparse
import sys

DEFAULTS = dict(
    reprompt_template="{prompt}{solution}{incorrect_attempt}{feedback}\n\nUse the context above to solve the original question correctly. If an incorrect attempt or feedback is shown, avoid the identified error. Please reason step by step before generating your final answer.",
    solution_template="\n\n# Successful reference solution:\n\n{successful_previous_attempt}",
    student_reasoning_suffix="\n\nPlease reason step by step before generating your final answer.",
)
SIPO_CTX = dict(
    reprompt_template="{solution}{incorrect_attempt}{feedback}{prompt}",
    solution_template="# Reference solution\n\nA correct solution to the question below, from an earlier attempt at it:\n\n{successful_previous_attempt}\n\n---\n\n",
    student_reasoning_suffix="",
)
# data/format/prompts.py PROMPT, plus the suffix data/preprocess.py:87 appends.
PROBLEM = "Let $x+1=2$. Find $x$."
SOLUTION = "Subtract 1 from both sides: $x = 1$.\n\n\\boxed{1}"


def common_suffix(a: str, b: str) -> str:
    n = 0
    while n < min(len(a), len(b)) and a[-1 - n] == b[-1 - n]:
        n += 1
    return a[len(a) - n:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--sipo-ctx", action="store_true")
    ap.add_argument("--enable-thinking", action="store_true",
                    help="user.yaml:18 sets enable_thinking false; pass this to check the other setting")
    a = ap.parse_args()
    cfg = SIPO_CTX if a.sipo_ctx else DEFAULTS

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    kw = dict(enable_thinking=a.enable_thinking)

    student_content = (f"{PROBLEM}\n\nPlease reason step by step, and put your final answer "
                       f"within \\boxed{{}}." + cfg["student_reasoning_suffix"])

    # ray_trainer.py:805 -- an empty suffix must not strip anything
    sfx = cfg["student_reasoning_suffix"]
    base = student_content
    if sfx and base.endswith(sfx):
        base = base[: -len(sfx)].rstrip()
    sol = cfg["solution_template"].format(successful_previous_attempt=SOLUTION)
    teacher_content = cfg["reprompt_template"].format(
        prompt=base, solution=sol, incorrect_attempt="", feedback="")

    s_txt = tok.apply_chat_template([{"role": "user", "content": student_content}],
                                    add_generation_prompt=True, tokenize=False, **kw)
    t_txt = tok.apply_chat_template([{"role": "user", "content": teacher_content}],
                                    add_generation_prompt=True, tokenize=False, **kw)
    s_ids = tok.apply_chat_template([{"role": "user", "content": student_content}],
                                    add_generation_prompt=True, tokenize=True, **kw)
    t_ids = tok.apply_chat_template([{"role": "user", "content": teacher_content}],
                                    add_generation_prompt=True, tokenize=True, **kw)

    print("=" * 76)
    print(f"mode: {'SIPO_CTX=1' if a.sipo_ctx else 'repo default'}   enable_thinking={a.enable_thinking}")
    print("=" * 76)
    print("--- ACTOR prompt (chat template applied) ---"); print(s_txt)
    print("--- TEACHER prompt (chat template applied) ---"); print(t_txt)

    suf = common_suffix(s_txt, t_txt)
    extra = t_txt[: len(t_txt) - len(suf)]
    s_head = s_txt[: len(s_txt) - len(suf)]

    print("=" * 76)
    ok = True

    # The whole point: everything from the student's own text to the generation prompt must
    # be byte-identical, so the teacher generates from the same position in the same format.
    c1 = student_content in t_txt and suf.endswith(s_txt[s_txt.index(student_content) + len(student_content):])
    print(f"[{'ok' if c1 else '!!'}] 学生内容之后的一切（含 generation prompt）完全相同")
    ok &= c1

    c2 = s_head == t_txt[: len(s_head)] if len(s_head) <= len(t_txt) else False
    print(f"[{'ok' if c2 else '!!'}] 两者共享同一个 chat-template 前缀")
    ok &= c2

    print(f"[  ] teacher 多出来的部分（应当只有特权信息块）:")
    # Diff the message CONTENT, not the templated text: longest-common-suffix can reach one
    # character past the block boundary and show a phantom leading blank line.
    body = (teacher_content[: -len(student_content)]
            if teacher_content.endswith(student_content) else teacher_content)
    for line in (body if body else "<无>").splitlines() or ["<空>"]:
        print(f"       | {line}")

    print(f"[  ] token 数  actor {len(s_ids)}  teacher {len(t_ids)}  差 {len(t_ids) - len(s_ids)}")
    if not a.sipo_ctx:
        print("[  ] 默认模板下两者的结尾指令不同，这是预期的；--sipo-ctx 才对齐")
    print("=" * 76)
    print("PASS" if ok or not a.sipo_ctx else "FAIL")
    return 0 if (ok or not a.sipo_ctx) else 1


if __name__ == "__main__":
    sys.exit(main())
