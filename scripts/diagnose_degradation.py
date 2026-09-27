#!/usr/bin/env python
"""Bin a SIPO run's console metrics over training steps and name what is degrading.

    python scripts/diagnose_degradation.py output/SIPO-reinforce-....log
    python scripts/diagnose_degradation.py output/a.log output/b.log   # compare arms

Reads verl's "step:N - key:value - ..." lines, so a saved log, a wandb export or
piped stdout all work. Validation lines are parsed separately -- they are the only
measurement taken on a fixed set of prompts.
"""
import re
import sys
from collections import defaultdict

NUM = re.compile(r"-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
# verl logs f"{k}:{pprint.pformat(v)}", and numpy >= 2 (NEP 51) renders reduced
# metrics as "np.float64(0.5)". Strip the constructor before matching, or its own
# digits ("64") get read as the value.
WRAPPER = re.compile(r"^\s*(?:[A-Za-z_][\w.]*\()+")


def extract(line, key):
    i = line.find(key + ":")
    if i < 0:
        return None
    chunk = line[i + len(key) + 1:]
    j = chunk.find(" - ")
    if j >= 0:
        chunk = chunk[:j]
    m = NUM.match(WRAPPER.sub("", chunk).strip())
    return float(m.group()) if m else None


# (label, key, format). None separates table sections.
PANELS = [
    ("退化的形态 -- 是能力掉了还是写不完", [
        ("reward",      "critic/rewards/mean",              "{:7.3f}"),
        ("acc(0/1)",    "reward_extra/acc/mean",            "{:7.3f}"),
        ("resp_len",    "response_length/mean",             "{:7.0f}"),
        ("trunc%",      "response_length/clip_ratio",       "{:7.3f}"),
        ("prompt_len",  "prompt_length/mean",               "{:7.0f}"),
    ]),
    ("策略健康 -- 熵塌了 A_r 就死了", [
        ("entropy",     "actor/entropy",                    "{:7.4f}"),
        ("grad_norm",   "actor/grad_norm",                  "{:7.4f}"),
        ("pg_clipfrac", "actor/pg_clipfrac",                "{:7.4f}"),
        ("A_r mean",    "critic/advantages/mean",           "{:7.4f}"),
        ("A_r max",     "critic/advantages/max",            "{:7.4f}"),
    ]),
    # The analytic arm drives the gradient through the divergence LOSS, not through A_d, so
    # opd/* is only descriptive there. scaled = beta * divergence, and what matters
    # is its size next to the reward half of the same loss.
    ("analytic arm -- 散度损失 vs 奖励损失", [
        ("reward_loss",   "sipo/reward_loss",                        "{:9.2e}"),
        ("divergence",    "sipo/divergence_loss",                    "{:9.2e}"),
        ("beta*div",      "sipo/scaled_divergence_loss",             "{:9.2e}"),
        ("total",         "sipo/total_loss",                         "{:9.2e}"),
    ]),
    # What the teacher actually saw, per rollout. "successful_reference" is real hindsight:
    # the target failed and the teacher was shown a sibling's correct solution.
    # "incorrect_reference" is the opposite -- the target SUCCEEDED and the teacher was shown
    # a sibling's wrong attempt, so it holds no answer the student lacks. ray_trainer.py:777
    # inserts those unconditionally; the SIPO repo has no such path, and the share below is
    # how much of the distilled mass is being pulled toward an uninformed teacher.
    ("teacher context 的构成", [
        ("有解组占比",  "self_distillation/success_group_fraction",         "{:7.3f}"),
        ("正确参考",    "teacher/ctx_successful_ref_frac",                  "{:7.3f}"),
        ("错误参考",    "teacher/ctx_incorrect_ref_frac",                   "{:7.3f}"),
        ("进 mask",     "self_distillation/reprompt_sample_fraction",       "{:7.3f}"),
    ]),
    ("A_d 是不是元凶", [
        ("A_d mean",    "opd/mean",                         "{:7.3f}"),
        ("A_d rms_clip","distill_adv/rms_clipped",          "{:7.3f}"),
        ("|A_r| mean",  "sipo/reward_adv_abs_mean",         "{:7.4f}"),
        ("clip_lo",     "distill_adv/clip_frac_low",        "{:7.3f}"),
        ("clip_hi",     "distill_adv/clip_frac_high",       "{:7.3f}"),
        ("tok_frac",    "sipo/teacher_token_frac",          "{:7.3f}"),
        ("A_tot mean",  "sipo/adv_mean",                    "{:7.4f}"),
    ]),
]
BETA_KEY = "sipo/divergence_beta"
ALL_KEYS = [k for _, rows in PANELS for _, k, _ in rows] + [BETA_KEY]
# The same metrics under the names logged before the 2026-09-23 rename, so older logs still
# parse. rms_clipped and clip_frac_* have no new name: they only exist in older logs.
OLD_NAMES = {
    "sipo/reward_loss": "self_instruction/reward_loss",
    "sipo/divergence_loss": "self_instruction/divergence_loss",
    "sipo/scaled_divergence_loss": "self_instruction/scaled_divergence_loss",
    "sipo/total_loss": "self_instruction/total_loss",
    "sipo/divergence_beta": "self_instruction/divergence_beta",
    "sipo/teacher_token_frac": "self_instruction/distill_token_fraction",
    "sipo/adv_mean": "self_instruction/adv_mean",
    "sipo/reward_adv_abs_mean": "distill_adv/reward_adv_abs_mean",
    "opd/mean": "distill_adv/mean",
    "teacher/ctx_successful_ref_frac": "self_distillation/successful_reference_fraction",
    "teacher/ctx_incorrect_ref_frac": "self_distillation/incorrect_reference_fraction",
}


def parse(path):
    train, val, beta = {}, {}, None
    lines = sys.stdin if path == "-" else open(path, errors="replace")
    for line in lines:
        if "step:" not in line:
            continue
        step = extract(line, "step")
        if step is None:
            continue
        step = int(step)
        got = {}
        for k in ALL_KEYS:
            v = extract(line, k)
            if v is None and k in OLD_NAMES:
                v = extract(line, OLD_NAMES[k])
            if v is not None:
                got[k] = v
        if got:
            train.setdefault(step, {}).update(got)
        if beta is None and BETA_KEY in got:
            beta = got[BETA_KEY]
        # ray_trainer.py:1093 picks the core variable per data source: "acc" when the
        # reward function reports it, "reward" otherwise. Matching only /acc/ silently
        # shows an empty table for any run whose scorer emits no acc key.
        for m in re.finditer(r"val-core/([\w\-.]+)/(\w+)/(?:mean|maj|best)[^:]*@(\d+)", line):
            v = extract(line, m.group(0))
            if v is not None:
                src, var = m.group(1), m.group(2)
                val.setdefault(step, {})[src if var == "acc" else f"{src}({var})"] = v
    return train, val, beta


def bins(steps, n=8):
    if not steps:
        return []
    lo, hi = min(steps), max(steps)
    if hi == lo:
        return [(lo, hi)]
    w = max(1, round((hi - lo + 1) / n))
    return [(s, min(s + w - 1, hi)) for s in range(lo, hi + 1, w)]


def report(path):
    train, val, beta = parse(path)
    if not train:
        print(f"!! {path}: 没解析到任何 step 行")
        return
    steps = sorted(train)
    print(f"\n{'=' * 78}\n{path}\n  steps {min(steps)}..{max(steps)}  "
          f"({len(steps)} 个采样点)   beta = {beta if beta is not None else '未记录'}\n{'=' * 78}")

    bs = bins(steps)

    def avg(key, a, b):
        vs = [train[s][key] for s in steps if a <= s <= b and key in train[s]]
        return sum(vs) / len(vs) if vs else None

    for title, rows in PANELS:
        present = [r for r in rows if any(r[1] in train[s] for s in steps)]
        if not present:
            continue
        print(f"\n-- {title}")
        print(f"  {'step':>12} " + " ".join(f"{lab:>11}" for lab, _, _ in present))
        for a, b in bs:
            cells = []
            for _, key, fmt in present:
                v = avg(key, a, b)
                cells.append(f"{fmt.format(v):>11}" if v is not None else f"{'-':>11}")
            print(f"  {f'{a}-{b}':>12} " + " ".join(cells))

    if val:
        print("\n-- 固定验证集 (唯一不受 prompt 抽样影响的信号)")
        names = sorted({n for d in val.values() for n in d})
        print(f"  {'step':>12} " + " ".join(f"{n[:11]:>11}" for n in names))
        for s in sorted(val):
            print(f"  {s:>12} " + " ".join(
                f"{val[s][n]:>11.3f}" if n in val[s] else f"{'-':>11}" for n in names))

    verdict(train, val, steps, beta)


def verdict(train, val, steps, beta):
    print("\n-- 判读")
    head, tail = steps[:max(1, len(steps) // 5)], steps[-max(1, len(steps) // 5):]

    def delta(key):
        h = [train[s][key] for s in head if key in train[s]]
        t = [train[s][key] for s in tail if key in train[s]]
        if not h or not t:
            return None, None, None
        h, t = sum(h) / len(h), sum(t) / len(t)
        return h, t, t - h

    def line(ok, msg):
        print(f"  [{'  ' if ok is None else 'ok' if ok else '!!'}] {msg}")

    r0, r1, dr = delta("critic/rewards/mean")
    if dr is not None:
        line(dr > -0.05, f"reward {r0:.3f} -> {r1:.3f}  ({dr:+.3f})")

    c0, c1, dc = delta("response_length/clip_ratio")
    if dc is not None:
        line(dc < 0.05,
             f"截断率 {c0:.3f} -> {c1:.3f}  ({dc:+.3f})"
             + ("   <- 退化很大程度是写不完，先加长度" if dc > 0.05 else ""))
    else:
        line(None, "没有 response_length/clip_ratio,无法区分「答错」和「没写完」")

    # Entropy and A_r are judged on the RATIO, not a floor. Halving either one is
    # already fatal to GRPO: identical rollouts give a zero group std, hence A_r = 0.
    e0, e1, de = delta("actor/entropy")
    if e1 is not None and e0:
        line(e1 > 0.05 and e1 / e0 > 0.5,
             f"熵 {e0:.4f} -> {e1:.4f}  ({e1 / e0:.2f}x)"
             + ("   <- 正在塌。8 条 rollout 趋同则组内 std -> 0,A_r 消失,"
                "剩下 beta*A_d 独自驱动梯度" if e1 <= 0.05 or e1 / e0 <= 0.5 else ""))

    a0, a1, _ = delta("critic/advantages/max")
    if a1 is not None and a0:
        line(a1 > 0.1 and a1 / a0 > 0.5,
             f"A_r 幅度 max {a0:.4f} -> {a1:.4f}  ({a1 / a0:.2f}x)"
             + ("   <- 奖励信号在失去梯度" if a1 <= 0.1 or a1 / a0 <= 0.5 else ""))

    # A_d only lands on reprompt-masked tokens; A_r lands on all of them. Report both
    # ends: this ratio climbs on its own as entropy collapse shrinks |A_r|, so A_d can
    # end up dominating without beta ever changing.
    rms = delta("distill_adv/rms_clipped")
    rabs = delta("sipo/reward_adv_abs_mean")
    frac = delta("sipo/teacher_token_frac")[1] or 1.0
    if rms[1] is not None and rabs[0] and rabs[1] and beta is not None:
        r_head = beta * rms[0] * frac / rabs[0]
        r_tail = beta * rms[1] * frac / rabs[1]
        line(r_tail < 1.0,
             f"beta*A_d / |A_r| = {r_head:.2f} -> {r_tail:.2f}"
             + ("   <- A_d 已盖过奖励信号" if r_tail >= 1.0
                else "   <- 在爬,A_d 占比越来越大" if r_tail > 1.5 * r_head else ""))

    m0, m1, _ = delta("opd/mean")
    if m1 is not None:
        line(abs(m1) < 0.5,
             f"A_d mean = {m1:+.3f}"
             + ("   <- 基线没扣干净。E[A_d] = -KL <= 0,残留的负均值"
                "会无差别压低所有 reprompt token 的 logp" if abs(m1) >= 0.5 else ""))

    # Do NOT ratio the two loss VALUES. On-policy the PPO surrogate has ratio == 1 and GRPO
    # advantages are centred, so its value sits at ~0 by construction while its gradient is
    # the full policy gradient; a KL is strictly positive. The ratio would read as tens of
    # thousands at any beta and means nothing. grad_norm is the comparable quantity: run
    # beta=0 and see how much of the gradient the divergence term adds on top.
    sdl = delta("sipo/scaled_divergence_loss")
    if sdl[1] is not None:
        gn = delta("actor/grad_norm")
        line(None, f"beta*散度损失 = {sdl[0]:.2e} -> {sdl[1]:.2e}  (损失数值，不是梯度占比)")
        if gn[1] is not None:
            line(None, f"grad_norm {gn[0]:.4f} -> {gn[1]:.4f}  <- 和 beta=0 的同期值相比，"
                       "超出的部分就是散度项贡献的梯度")

    if val:
        vs = sorted(val)
        if len(vs) >= 2:
            f = lambda s: sum(val[s].values()) / len(val[s])
            line(f(vs[-1]) >= f(vs[0]) - 0.02,
                 f"固定验证集均值 {f(vs[0]):.3f} (step {vs[0]}) -> {f(vs[-1]):.3f} (step {vs[-1]})")
    else:
        line(None, "日志里没有 val-core/*,固定集信号缺失 -- 训练 reward 无法单独判定能力变化")

    if beta is not None and beta > 0:
        line(None, "beta > 0 且没有 beta=0 对照,以上都无法归因到 A_d")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for p in sys.argv[1:]:
        report(p)
