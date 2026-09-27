#!/bin/bash
#
# SIPO: analytic reverse-KL divergence  vs.  REINFORCE-form distillation advantage
# Single node, 8 x A100 40GB.
#
# Usage:
#   bash experiments/reinforce_adv/run_sipo_reinforce_adv.sh <mode> [--dry-run]
#
#   mode = diag       10 steps of arm B with SIPO_LAMBDA_START=0, so A_d is measured but
#                     the advantage is untouched: plain GRPO with the teacher's numbers logged.
#          reinforce  arm B, RLSD (arXiv:2604.03128):
#                       A_t = A_r * ((1-lam) + lam * clip(exp(sign(A_r) * A_d), 1-eps, 1+eps))
#                     lam: SIPO_LAMBDA_START -> 0 over SIPO_LAMBDA_STEPS. Every row's teacher
#                     sees the dataset answer (SIPO_TEACHER_CTX=gold) and is a periodic snapshot of
#                     the student (TEACHER=periodic, TEACHER_SYNC=10).
#          analytic   arm A baseline: analytic reverse KL over top-100 + tail (alpha=1).
#
# Env:
#
# Recommended combinations:
#   controlled A/B at full length : PROFILE=full SP=2      (both arms; run diag SP=2 first)
#   controlled A/B, fastest       : PROFILE=ab             (free for PRESET=gen)
#   arm B alone, longest+fastest  : PROFILE=full FUSED=1   (SP=1)
#
#   PRESET   gen  -> SciKnowEval / ToolUse, scalar reward   (default; DATA_PATH=datasets/tooluse)
#            rich -> LiveCodeBench v6 with execution feedback (DATA_PATH=datasets/lcb_v6)
#            math -> DAPO-Math-17k. Scalar reward like `gen` (no environment feedback),
#                    but with DAPO's reward setup: its overlong buffer replaces the hard
#                    zero at the response cap with a graded penalty. The algorithm stays
#                    GRPO -- none of DAPO's clip-higher / dynamic-sampling changes.
#   PROFILE  ab   -> lengths both arms can run; use this for the controlled A/B (default)
#            full -> full stock lengths; requires FUSED=1, i.e. mode=reinforce only
#   FUSED    1    -> actor_rollout_ref.model.use_fused_kernels=True. Skips materialising the
#                    (L x 151936) logits tensor. INCOMPATIBLE with mode=analytic, which needs
#                    top-k logits (dp_actor.py:207-208 raises).
#   SP       ulysses_sequence_parallel_size (default 1; 2 or 4 to keep length in mode=analytic)
#   ROLLOUT  rollout engine, vllm (default) or sglang -- see the sglang warning below
#   OVERLONG=1    PRESET=math only: opt into DAPO's length penalty. Off by default so
#                 the reward matches the gen and rich presets exactly.
#   EVAL_SUITE=1  validate on the 6-benchmark math suite instead of the task test split
#   EVAL_N        rollouts per eval problem (default 16); EVAL_DIR where the suite lives
#   STEPS    trainer.total_training_steps. REQUIRED for sweeps: without it a run does
#            30 epochs (3780 steps on tooluse). diag ignores it, it always does 10.
#   DATA_PATH / MODEL / BETA / SEED / LR   individual overrides (BETA: analytic arm only)

set -euo pipefail

MODE="${1:-}"
case "$MODE" in
    diag|analytic|reinforce) shift ;;
    *) echo "Usage: $0 {diag|analytic|reinforce} [--dry-run] [hydra.overrides=...]"; exit 1 ;;
esac

# Anything else is forwarded to the trainer as a hydra override, so one-off settings do
# not need an env knob. Appended last, which is what hydra takes as the winner.
DRY_RUN=false
EXTRA=()
for arg in "$@"; do
    if [[ "$arg" == "--dry-run" ]]; then
        DRY_RUN=true
    else
        EXTRA+=("$arg")
    fi
done

PRESET="${PRESET:-gen}"
REWARD_ARGS=""
EVAL_ARGS=""
PROFILE="${PROFILE:-ab}"
FUSED="${FUSED:-0}"
# Card size. Every default below was picked for a 40GB A100, where the (seq x 152k) logit
# tensor and the activation stack are the binding constraints. An 80GB card removes both,
# and the levers that bought the memory -- Ulysses SP, rollout TP, a micro batch of one --
# all cost throughput, so they should come back off. Detect rather than ask; VRAM_GB
# overrides, and an unreadable nvidia-smi keeps the 40GB behaviour.
# Detection must not be able to fail the run. This file is `set -euo pipefail`, so a
# command substitution holding a pipeline whose first stage is missing takes the whole
# script down with status 127 -- and with nvidia-smi's stderr sent to /dev/null, bash's
# own "command not found" goes with it, leaving no output at all. Probe for the binary
# first, keep the pipeline out of the assignment, and only accept digits.
if [ -z "${VRAM_GB:-}" ]; then
    VRAM_GB=40
    if command -v nvidia-smi >/dev/null 2>&1; then
        _vram_mib="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null \
                     | head -1 | tr -d '[:space:]' || true)"
        case "${_vram_mib:-}" in
            ''|*[!0-9]*) ;;                      # unreadable -> keep the 40GB defaults
            *) VRAM_GB=$(( _vram_mib / 1024 )) ;;
        esac
        unset _vram_mib
    fi
fi
if [ "$VRAM_GB" -ge 70 ]; then BIG_CARD=1; else BIG_CARD=0; fi

SP="${SP:-1}"
# Rollout engine. user.yaml pins vllm, which is what every script here assumes.
# ROLLOUT=sglang works (base.py registers it, and rollout.mode=async is the default) but
# see the WARNING below before using it.
ROLLOUT="${ROLLOUT:-vllm}"

# Rollout TP. At full length generation is KV-bound, and a TP group amortises the 16.4GB of
# bf16 weights over its GPUs, leaving more of the 0.75 budget for KV -> more concurrency.
# At the short "ab" lengths TP=1 avoids the per-layer all-reduce and is faster.
# Rollout TP, and it is not only about KV. rollout.mode defaults to async, and the async
# server hardcodes sleep(level=1) (vllm_async_server.py:572), which frees the KV cache but
# NOT the weights -- unlike the sync path, which uses VLLM_SLEEP_LEVEL=2 on vllm >= 0.8.5.
# So vLLM's 16.4GB of bf16 weights, divided by TP, stays resident through update_policy and
# is subtracted from the training budget. TP=2 halves that to 7.6GB per card, which is worth
# more than the per-layer all-reduce it costs, on 80GB as much as on 40GB.
if [[ "$PROFILE" == "full" ]]; then TP="${TP:-2}"; else TP="${TP:-1}"; fi

# ---- guards on combinations that cannot work -------------------------------
if [[ "$FUSED" == "1" && "$MODE" == "analytic" ]]; then
    echo "ERROR: FUSED=1 is incompatible with mode=analytic. That arm requests top-k logits"
    echo "       and dp_actor.py:207 raises 'Logit distillation requires disabling fused kernels'."
    exit 1
fi
if [[ "$BIG_CARD" != "1" && "$PROFILE" == "full" && "$FUSED" != "1" && "$SP" == "1" ]]; then
    echo "ERROR: PROFILE=full needs one of the two length levers on a 40GB card:"
    echo "         FUSED=1  -- free, but mode=reinforce only"
    echo "         SP=2|4   -- Ulysses sequence parallelism, works for BOTH arms"
    echo "       For a controlled A/B at full length use PROFILE=full SP=2 on both arms."
    exit 1
fi

# =============================================================================
# PATHS -- user.yaml hard-codes /path/to/SIPO; override it all.
# =============================================================================
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export USER="${USER:-$(whoami)}"
mkdir -p "$PROJECT_ROOT/output"

# EVAL_SUITE=1 validates on AIME24 / AIME25 / AMC23 / MATH-500 / Minerva / OlympiadBench
# instead of the task's own test split. verl keys validation metrics on data_source, so
# each lands under its own val-core/<name>/ prefix. Build them first with
# `bash data/build_math_eval_suite.sh`.
#   EVAL_N     val_kwargs.n. Defaults to 1 because build_math_eval_suite.sh already bakes
#              avg@k in by duplicating rows per benchmark -- verl applies val_kwargs.n
#              uniformly across val files, so duplication is the only way to sample AIME
#              32x without also sampling OlympiadBench 32x. Raising EVAL_N multiplies on
#              top of the built-in repeats.
#   EVAL_DIR   where build_math_eval_suite.sh put them.
if [ "${EVAL_SUITE:-0}" = "1" ]; then
    EVAL_DIR="${EVAL_DIR:-$PROJECT_ROOT/datasets/math_eval}"
    EVAL_N="${EVAL_N:-1}"
    _val=""
    for name in aime24 aime25 amc23 math500 minervamath olympiadbench; do
        [ -n "$_val" ] && _val="$_val,"
        _val="$_val$EVAL_DIR/$name/test.parquet"
    done
    EVAL_ARGS="data.val_files=[$_val] actor_rollout_ref.rollout.val_kwargs.n=$EVAL_N"
fi

PATH_ARGS="vars.dir=$PROJECT_ROOT \
vars.log_dir=$PROJECT_ROOT/output \
vars.ckpt_dir=$PROJECT_ROOT/checkpoints \
custom_reward_function.path=$PROJECT_ROOT/verl/utils/reward_score/feedback/__init__.py"

# =============================================================================
# TASK PRESET
# =============================================================================
case "$PRESET" in
    gen)
        # Scalar reward, no environment feedback. tooluse prompts are p95 ~1.1k tokens
        # and answers p95 ~150 tokens, so the stock 8192 response budget is pure waste --
        # the "ab" lengths below are right-sizing, not a compromise.
        DATA_PATH="${DATA_PATH:-datasets/tooluse}"
        FEEDBACK_ARGS="actor_rollout_ref.actor.self_distillation.include_environment_feedback=False"
        if [[ "$PROFILE" == "full" ]]; then
            L_PROMPT=2048; L_RESP=8192; L_REPROMPT=10240; L_MODEL=18944
        else
            L_PROMPT=1536; L_RESP=2048; L_REPROMPT=4096;  L_MODEL=6144
        fi
        ;;
    rich)
        # LCBv6: teacher prompt carries the failing-test feedback, so length is load
        # bearing here. PROFILE=ab genuinely costs coverage of long problems.
        DATA_PATH="${DATA_PATH:-datasets/lcb_v6}"
        FEEDBACK_ARGS="actor_rollout_ref.actor.self_distillation.include_environment_feedback=True \
actor_rollout_ref.actor.self_distillation.environment_feedback_only_without_solution=False \
actor_rollout_ref.actor.self_distillation.teacher_update_rate=0.01"
        if [[ "$PROFILE" == "full" ]]; then
            L_PROMPT=2048; L_RESP=8192; L_REPROMPT=10240; L_MODEL=18944
        else
            L_PROMPT=2048; L_RESP=4096; L_REPROMPT=8192;  L_MODEL=12288
        fi
        ;;
    math)
        DATA_PATH="${DATA_PATH:-datasets/dapo_math}"
        # Scalar-reward setting, same as `gen`. The math verifier only emits feedback for
        # format errors and truncation anyway -- feedback/math.py has
        # correctness_feedback=False and the dispatch does not override it.
        FEEDBACK_ARGS="actor_rollout_ref.actor.self_distillation.include_environment_feedback=False"
        if [[ "$PROFILE" == "full" ]]; then
            L_PROMPT=2048; L_RESP=8192; L_REPROMPT=10240; L_MODEL=18944
        else
            L_PROMPT=2048; L_RESP=3072; L_REPROMPT=5120;  L_MODEL=8192
        fi
        # DAPO's reward half: keep our math verifier for scoring (DAPORewardManager takes
        # compute_score, and reward.py:117 feeds it the custom_reward_function), and add
        # the overlong buffer -- no penalty below max_resp_len - len, then a linear ramp to
        # -penalty_factor at the cap (dapo.py:116-121). That gives length pressure *before*
        # a response degenerates into a truncated zero-reward one, which is how the tooluse
        # run collapsed. Buffer is half the response budget, as in DAPO's own script.
        # Reward is a bare 0/1 verifier by default, the same as the gen and rich presets,
        # which set no reward args at all and so use the naive manager. Alignment here
        # matters more than borrowing DAPO's shaping: GRPO computes A_r on the shaped
        # reward, so a length penalty would flow into A_r and then sum with beta*A_d,
        # putting three signals in the advantage and making the math runs incomparable
        # with the others.
        #
        # OVERLONG=1 opts back into DAPO's overlong buffer: no penalty below
        # max_resp_len - len, then a linear ramp to -penalty_factor at the cap
        # (verl/workers/reward_manager/dapo.py:116-121).
        OVERLONG_LEN=$(( L_RESP / 2 ))
        if [ "${OVERLONG:-0}" = "1" ]; then
            REWARD_ARGS="reward_model.reward_manager=dapo \
+reward_model.reward_kwargs.max_resp_len=$L_RESP \
+reward_model.reward_kwargs.overlong_buffer_cfg.enable=True \
+reward_model.reward_kwargs.overlong_buffer_cfg.len=$OVERLONG_LEN \
+reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
+reward_model.reward_kwargs.overlong_buffer_cfg.log=True"
        fi
        ;;
    *) echo "ERROR: PRESET must be gen, rich or math"; exit 1 ;;
esac

# =============================================================================
# EXPERIMENT
# =============================================================================
# -----------------------------------------------------------------------------
# RESP=<n> overrides the response budget and re-derives the two lengths that must
# track it. Do not set L_REPROMPT / L_MODEL by hand.
#
# The teacher context is the original prompt plus ONE FULL peer response -- either a
# correct sibling rollout (solution_strs) or an incorrect one (incorrect_attempt_strs),
# both sliced out of response_texts in ray_trainer.py:759-791. So the teacher prompt is
# prompt + response, and the teacher SEQUENCE is that plus the response being scored:
#
#     L_REPROMPT = L_PROMPT + L_RESP
#     L_MODEL    = L_REPROMPT + L_RESP + 512     (= 2*L_RESP + L_PROMPT + 512)
#
# Undersizing L_REPROMPT does not raise. ray_trainer.py:852-862 calls
# apply_chat_template(truncation=True, max_length=max_reprompt_len), which truncates
# from the RIGHT -- taking out the tail of the hindsight section and then the
# add_generation_prompt suffix. The teacher then conditions on a mangled context and
# A_d quietly measures the wrong thing. There is no warning for this.
#
# Memory is not the binding constraint: the teacher pass runs under torch.no_grad
# (dp_actor.py:852), rollout.free_cache_engine returns vLLM's pool during the update,
# and use_fused_kernels avoids materialising L_MODEL x 151936 logits. Measured shape of
# the peak, per GPU on a 40GB card, Qwen3-8B, micro_bsz=1, grad-ckpt on:
#     RESP= 8192 -> 11.5 GB    RESP=16384 -> 16.2 GB
#     RESP=12288 -> 13.9 GB
# What actually scales is wall clock, roughly 1.5-1.6x per doubling.
#
# The ceiling is not memory but rope: Qwen3-8B has max_position_embeddings=40960 and
# no rope_scaling, and L_MODEL = 2*L_RESP + L_PROMPT + 512, so RESP=16384 (-> 35328)
# is the largest power-of-two budget that fits. RESP=19456 already overflows. The
# check below reads the model's own config rather than assuming Qwen3-8B.
L_RESP_DEFAULT=$L_RESP
if [ -n "${RESP:-}" ]; then
    L_RESP=$RESP
    L_REPROMPT=$(( L_PROMPT + L_RESP ))
    L_MODEL=$(( L_REPROMPT + L_RESP + 512 ))
    # The math preset already baked the old L_RESP into REWARD_ARGS, so rebuild it.
    if [ -n "${REWARD_ARGS:-}" ]; then
        OVERLONG_LEN=$(( L_RESP / 2 ))
        REWARD_ARGS="reward_model.reward_manager=dapo \
+reward_model.reward_kwargs.max_resp_len=$L_RESP \
+reward_model.reward_kwargs.overlong_buffer_cfg.enable=True \
+reward_model.reward_kwargs.overlong_buffer_cfg.len=$OVERLONG_LEN \
+reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
+reward_model.reward_kwargs.overlong_buffer_cfg.log=True"
    fi
fi

MODEL="${MODEL:-Qwen/Qwen3-8B}"

# L_MODEL has to fit the model's positional range. vLLM does catch this, but only
# after loading weights, several minutes in. Read config.json directly -- conda is not
# activated until verl_training.sh, so transformers is not importable here.
MAX_POS="$(python3 - "$MODEL" <<'PY' 2>/dev/null || true
import glob, json, os, sys
m = sys.argv[1]
cand = [os.path.join(m, "config.json")] + glob.glob(os.path.join(
    os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
    "hub", "models--" + m.replace("/", "--"), "snapshots", "*", "config.json"))
for c in cand:
    if os.path.isfile(c):
        print(json.load(open(c)).get("max_position_embeddings", 0)); break
PY
)"
if [ -n "${MAX_POS:-}" ] && [ "$MAX_POS" -gt 0 ] 2>/dev/null; then
    if [ "$L_MODEL" -gt "$MAX_POS" ]; then
        MAX_RESP=$(( (MAX_POS - L_PROMPT - 512) / 2 ))
        echo "ERROR: max_model_len=$L_MODEL exceeds $MODEL's max_position_embeddings=$MAX_POS." >&2
        echo "       L_MODEL = 2*L_RESP + L_PROMPT + 512 because the teacher context carries" >&2
        echo "       a full peer response. With L_PROMPT=$L_PROMPT the cap is RESP=$MAX_RESP." >&2
        exit 1
    fi
else
    echo "  NOTE: could not read max_position_embeddings for $MODEL; not checking L_MODEL=$L_MODEL." >&2
fi
BETA="${BETA:-0.1}"
SEED="${SEED:-42}"
# Teacher regularization, matching the SIPO repo this work is based on: its
# SelfDistillationConfig validates teacher_regularization against exactly
# ["ema", "trust-region"] and raises on anything else, so "student" -- the live actor read
# under the augmented prompt -- is not a configuration that repo can express. Its default
# is ema at 0.05 (only the LCBv6 script lowers it to 0.01). SDPO's Table 4 measures the
# unregularized variant directly: q_theta 36.1/29.8 best/avg to step 90 against 49.3/45.3
# for EMA, captioned "Training of the q_theta eventually diverges."
#   ema           theta' <- (1-a) theta' + a theta
#   trust-region  q ~ exp((1-a) log q_ref + a log q_theta); needs use_fused_kernels=False
#   student       unregularized; what runs before 2026-09-13 used, kept for the ablation
# Divergence shape: 0 = forward KL (mass covering), 1 = reverse KL (mode seeking),
# 0.5 = the Jensen-Shannon mixture. Only the analytic arm can vary it -- the reinforce arm
# is the single-sample score-function estimator of reverse KL specifically, so its config
# validation rejects anything but 1.0.
ALPHA="${ALPHA:-1.0}"
# SIPO_CTX=1 fixes what the teacher is conditioned on. Two semantic changes and one
# layout change.
#
# Semantics, matching the SIPO repo:
#   - a rollout that SUCCEEDED currently gets a sibling's INCORRECT attempt as context,
#     i.e. a teacher holding no answer the student lacks. It gets none, and falls through
#     to a successful sibling's solution if the group has one.
#   - the teacher prompt has the student's "reason step by step" suffix stripped
#     (ray_trainer.py:834). An empty suffix makes that strip a no-op.
#
# Layout. The divergence is only meaningful if the two prompts differ by the privileged
# information and nothing else. SIPO's own template appends the hindsight and then a
# closing "Correctly solve the original question.", so the teacher's last instruction is
# not the student's -- on math the student ends with "put your final answer within
# \boxed{}" and the teacher does not, which shifts the teacher's distribution for reasons
# that have nothing to do with hindsight. Putting the privileged block FIRST and the
# original prompt LAST makes the teacher prompt end with the student prompt verbatim, so
# the two differ by a prepended, clearly delimited block and by nothing else. Each block
# carries a heading naming what it is and closes with a --- rule.
SIPO_CTX="${SIPO_CTX:-0}"
CTX_ARGS=()
if [ "$SIPO_CTX" = "1" ]; then
    # Two layers of quoting. $'...' is the shell's, turning \n into real newline bytes.
    # The inner single quotes are hydra's: its override grammar reads a bare value with
    # its own lexer, where "{" opens a dict literal and a raw newline has no valid token,
    # so an unquoted template dies with LexerNoViableAltException. Escaping as the two
    # characters \n parses but stores a literal backslash-n, a different string.
    _K="actor_rollout_ref.actor.self_distillation"
    CTX_ARGS=(
        "$_K.include_incorrect_attempt=False"
        "$_K.student_reasoning_suffix=''"
        "$_K.reprompt_template='{solution}{incorrect_attempt}{feedback}{prompt}'"
        "$_K.solution_template='"$'# Reference solution\n\nA correct solution to the question below, from an earlier attempt at it:\n\n{successful_previous_attempt}\n\n---\n\n'"'"
        "$_K.feedback_template='"$'# Feedback on an earlier attempt\n\n{feedback_raw}\n\n---\n\n'"'"
        # Unused while include_incorrect_attempt is False, set so the family stays coherent.
        "$_K.incorrect_attempt_template='"$'# Incorrect earlier attempt\n\nThis answer to the question below was wrong. Do not repeat its mistake:\n\n{incorrect_attempt}\n\n---\n\n'"'"
    )
    unset _K
fi

# RLSD knobs (mode=reinforce and diag). Defaults are the paper's: eps 0.2, lambda 0.5 -> 0
# over 50 steps, dataset answer as the teacher's context for every row.
#   SIPO_TEACHER_CTX=gold  every row's teacher sees the final answer (paper)
#   SIPO_TEACHER_CTX=sipo  keep the SIPO context instead: a successful sibling rollout where one exists
SIPO_EPS="${SIPO_EPS:-0.2}"
SIPO_LAMBDA_START="${SIPO_LAMBDA_START:-0.5}"
SIPO_LAMBDA_STEPS="${SIPO_LAMBDA_STEPS:-50}"
SIPO_TEACHER_CTX="${SIPO_TEACHER_CTX:-gold}"
# SIPO_BASELINE=confidence builds the weight from the evidence's per-token advantage (A_d minus
# its mean at the student's confidence level) instead of the raw log-ratio, whose expectation
# -KL_t makes it a push against exploration. SIPO_NEG_ONLY=1 reweights wrong rollouts only.
SIPO_BASELINE="${SIPO_BASELINE:-none}"
SIPO_NEG_ONLY="${SIPO_NEG_ONLY:-0}"
# SIPO_EVIDENCE=contrastive: e = log pi(y|x,gold) - log pi(y|x,wrong answer) instead of the published
# log pi(y|x,gold) - log pi_S(y|x). Both sides answer-conditioned -> the "be decisive" direction
# cancels; only answer-specific tokens are reweighted. One extra teacher forward per micro batch.
SIPO_EVIDENCE="${SIPO_EVIDENCE:-opd}"
# CONTRAST_NEG=own|mode: which wrong answer the second teacher sees. own = each wrong rollout's own
# boxed answer (correct/truncated rows: the group's most common wrong answer); mode = group mode for all.
CONTRAST_NEG="${CONTRAST_NEG:-own}"
# CONTRAST_CTX=answer|trajectory|auto: answer = gold answer vs wrong answer in the gold block (math,
# tooluse); trajectory = successful sibling rollout vs failed sibling rollout under the same solution
# template (code: the ground truth is a test suite, not an answer), rows need both siblings; auto =
# trajectory where both siblings exist, else the answer pair (science: keeps the all-wrong groups).
CONTRAST_CTX="${CONTRAST_CTX:-answer}"
# CONTRAST_UNJUDGED=skip|mode: a wrong rollout without its own valid judged answer gets no teacher term
# (skip) or the group's most common wrong answer (mode, the behaviour of runs before 2026-09-23).
CONTRAST_UNJUDGED="${CONTRAST_UNJUDGED:-skip}"
# IMPUTE=1: all-wrong groups get A = -IMPUTE_WRONG_SCALE * c instead of GRPO's 0 (c = batch mean |A|
# of the wrong rollouts in mixed groups); IMPUTE_CORRECT_SCALE > 0 gives all-correct groups +scale*c.
# The RLSD reweighting then acts ONLY on those rows; mixed groups stay bit-identical to GRPO.
# Pair with SIPO_LAMBDA_STEPS=0 so the reweighting on the imputed rows does not fade out.
IMPUTE="${IMPUTE:-0}"
IMPUTE_WRONG_SCALE="${IMPUTE_WRONG_SCALE:-1.0}"
IMPUTE_CORRECT_SCALE="${IMPUTE_CORRECT_SCALE:-0.0}"
# ALLWRONG_KAPPA>0: additive contrastive credit on all-wrong groups, A_t = ALLWRONG_KAPPA*clip(e_t, +-eps),
# a constant. Scale: a wrong rollout in a mixed group gets lam*|A_r|*e_t from the teacher and all_wrong/c
# logs the mean |A_r| of those rollouts (~0.3 early), so 0.15 = lam*c is the same teacher term; keep
# ALLWRONG_KAPPA*SIPO_EPS below 1/8. Needs SIPO_EVIDENCE=contrastive. Mixed groups keep the multiplicative
# form; all-correct groups stay at 0.
ALLWRONG_KAPPA="${ALLWRONG_KAPPA:-0}"
# SIPO_FORM=multiplicative|additive. additive: one formula for every row, A_t = A_r + SIPO_KAPPA*clip(e_t, +-eps)
# (all-wrong groups fall out with A_r = 0), sign-preserving clamp on mixed rows (inactive while
# SIPO_KAPPA*SIPO_EPS < 1/8). Needs SIPO_EVIDENCE=contrastive and ALLWRONG_KAPPA=0; lambda is unused.
SIPO_FORM="${SIPO_FORM:-multiplicative}"
SIPO_KAPPA="${SIPO_KAPPA:-0}"
if [ "$MODE" != "analytic" ]; then
    _K="actor_rollout_ref.actor.self_distillation"
    CTX_ARGS+=("$_K.sipo_eps=$SIPO_EPS"
               "$_K.sipo_lambda_start=$SIPO_LAMBDA_START"
               "$_K.sipo_lambda_steps=$SIPO_LAMBDA_STEPS"
               "$_K.sipo_baseline=$SIPO_BASELINE"
               "$_K.sipo_evidence=$SIPO_EVIDENCE"
               "$_K.sipo_contrast_negative=$CONTRAST_NEG"
               "$_K.sipo_contrast_context=$CONTRAST_CTX"
               "$_K.sipo_contrast_unjudged=$CONTRAST_UNJUDGED"
               "$_K.sipo_form=$SIPO_FORM"
               "$_K.sipo_kappa=$SIPO_KAPPA")
    [ "$SIPO_NEG_ONLY" = "1" ] && CTX_ARGS+=("$_K.sipo_negative_only=True")
    [ "$ALLWRONG_KAPPA" != "0" ] && CTX_ARGS+=("$_K.sipo_all_wrong_kappa=$ALLWRONG_KAPPA")
    if [ "$SIPO_FORM" = "additive" ]; then
        [ "$SIPO_EVIDENCE" = "contrastive" ] || { echo "ERROR: SIPO_FORM=additive needs SIPO_EVIDENCE=contrastive" >&2; exit 1; }
        [ "$ALLWRONG_KAPPA" = "0" ] || { echo "ERROR: SIPO_FORM=additive covers all-wrong groups itself; set ALLWRONG_KAPPA=0" >&2; exit 1; }
        [ "$SIPO_KAPPA" != "0" ] || { echo "ERROR: SIPO_FORM=additive needs SIPO_KAPPA>0" >&2; exit 1; }
    fi
    if [ "$IMPUTE" = "1" ]; then
        CTX_ARGS+=("$_K.impute_group_advantage=True"
                   "$_K.impute_wrong_scale=$IMPUTE_WRONG_SCALE"
                   "$_K.impute_correct_scale=$IMPUTE_CORRECT_SCALE")
    fi
    case "$SIPO_TEACHER_CTX" in
        gold) CTX_ARGS+=("$_K.gold_answer_for_all=True") ;;
        sipo) ;;
        *) echo "ERROR: SIPO_TEACHER_CTX must be gold or sipo, got '$SIPO_TEACHER_CTX'" >&2; exit 1 ;;
    esac
    unset _K
fi

if [ "$MODE" = "analytic" ]; then
    TEACHER="${TEACHER:-ema}"
else
    TEACHER="${TEACHER:-periodic}"   # RLSD: student snapshot every TEACHER_SYNC steps, frozen between
fi
TEACHER_RATE="${TEACHER_RATE:-0.05}"
TEACHER_SYNC="${TEACHER_SYNC:-10}"
case "$TEACHER" in
    ema|student) ;;
    trust-region)
        if [[ "$FUSED" == "1" ]]; then
            echo "ERROR: TEACHER=trust-region needs use_fused_kernels=False, but FUSED=1." >&2
            echo "       TrustRegionTeacher returns interpolated logits (dp_actor.py:69) that the" >&2
            echo "       fused linear+CE path cannot consume. Use FUSED=0 with SP=2, or TEACHER=ema." >&2
            exit 1
        fi ;;
    periodic) ;;
    *) echo "ERROR: TEACHER must be ema, periodic, trust-region or student (got '$TEACHER')" >&2; exit 1 ;;
esac
# ema and trust-region promote the actor worker to Role.ActorRolloutRef (main_ppo.py:135,186),
# so a second module stays resident; main_ppo.py:136 forbids that module from doubling as the
# KL reference, and use_kl_loss / use_kl_in_reward are both False in user.yaml:25,44.
LR="${LR:-1e-6}"
# Step cap. Without one, `reinforce` inherits user.yaml's total_epochs=30, which on
# tooluse is 4046/32 = 126 steps per epoch => 3780 steps. Far too long to sweep; set
# STEPS from the per-step wall clock in the diag log.
STEPS="${STEPS:-}"

# INVARIANT: ppo_mini_batch_size == train_batch_size keeps the run strictly on-policy
# (it is multiplied by rollout.n in fsdp_workers.py:242, so 32*8 = the whole batch =>
# one mini-batch; with ppo_epochs=1 that gives old_log_prob == log_prob => ratio == 1).
# Both arms rely on this; break it and A_d starts interacting with the PPO clip.
TRAIN_BATCH_SIZE=32
ROLLOUT_N=8

# =============================================================================
# MEMORY / SPEED
# =============================================================================
# vLLM sleeps during the training phase (rollout.free_cache_engine=True, sleep level
# from verl/third_party/vllm/__init__.py:33-49), so gpu_memory_utilization competes
# with the ROLLOUT peak, not the training peak. With param_offload on, the card is
# nearly free during rollout, so vLLM can be given a lot.
#
# teacher_regularization=student means the teacher IS the current actor
# (fsdp_workers.py:898 leaves teacher_module=None, dp_actor falls back to actor_module),
# so there is no second model resident and no ref-offload hazard. If you switch to
# teacher_regularization=ema or trust-region, do NOT set
# actor_rollout_ref.ref.fsdp_config.param_offload=True: the teacher forward runs inside
# update_policy, and update_actor only brings actor_module_fsdp / the optimizer back to
# GPU, so an offloaded ref leaves the teacher on CPU.
#
# enable_gradient_checkpointing is already True (model/hf_model.yaml:34) and entropy is
# off (entropy_coeff=0), so neither is a lever here.
# A regularized teacher keeps a second module resident for the whole step: ema and
# trust-region promote the actor worker to Role.ActorRolloutRef, and ref.fsdp_config
# .param_offload defaults to false (and has to -- the teacher forward runs inside
# update_policy, which only pulls the actor back to GPU). That is ~2 GB per card of
# sharded bf16 weights that vLLM's pool has to leave room for during rollout, so drop
# its share. TEACHER=student has no second module and keeps 0.75, which is what every
# run before 2026-09-13 used -- so a student ablation still reproduces them exactly.
if [ "$TEACHER" = "student" ]; then
    GPU_MEM="${GPU_MEM:-0.75}"
else
    GPU_MEM="${GPU_MEM:-0.70}"
fi

# With use_remove_padding the micro batch is a real packing factor, so widening it is the
# way to use an 80GB card -- but only for the reinforce arm. The analytic arm materialises
# the (seq x 152k) logits for both student and teacher, and those scale linearly with the
# micro batch: at 12k they are 23.1GB of a 29.0GB training peak at mb=1, so mb=2 alone
# takes it to 51.1GB and, with vLLM's resident weights on top, over the card. It stays at 1.
if [ "$BIG_CARD" = "1" ] && [ "$MODE" != "analytic" ]; then
    MICRO_BSZ="${MICRO_BSZ:-2}"
else
    MICRO_BSZ="${MICRO_BSZ:-1}"
fi

MEM_ARGS="trainer.n_gpus_per_node=8 \
trainer.nnodes=1 \
data.max_prompt_length=$L_PROMPT \
data.max_response_length=$L_RESP \
max_model_len=$L_MODEL \
actor_rollout_ref.actor.self_distillation.max_reprompt_len=$L_REPROMPT \
actor_rollout_ref.actor.ulysses_sequence_parallel_size=$SP \
actor_rollout_ref.actor.fsdp_config.param_offload=True \
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BSZ \
actor_rollout_ref.rollout.name=$ROLLOUT \
actor_rollout_ref.rollout.tensor_model_parallel_size=$TP \
actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEM"

# WARNING for ROLLOUT=sglang: sipo.yaml sets rollout.calculate_log_probs=True because
# algorithm.rollout_correction.rollout_is=token needs rollout_log_probs -- but
# ray_trainer.py:1936 applies that correction only `if "rollout_log_probs" in batch.batch`,
# so a backend that does not return them degrades SILENTLY rather than failing. If you
# switch, confirm the rollout-correction metrics are non-zero before trusting a run.

if [[ "$FUSED" == "1" ]]; then
    MEM_ARGS="$MEM_ARGS actor_rollout_ref.model.use_fused_kernels=True"
fi
# Speed levers, in order, once it fits:
#   - drop actor.fsdp_config.param_offload (worth ~10-20% step time; needs headroom)
#   - actor.use_dynamic_bsz=True actor.ppo_max_token_len_per_gpu=<budget>
#     CAUTION: prepare_dynamic_batch balances on the STUDENT attention_mask
#     (seqlen_balancing.py:374), and the teacher sequence is longer. Budget for
#     student+teacher, i.e. roughly half of what the student alone would allow.
# Memory levers, in order, if it OOMs:
#   - SP=2, then SP=4
#   - rollout.gpu_memory_utilization=0.60 / 0.45
#   - rollout.tensor_model_parallel_size=2

# =============================================================================
# ARM
# =============================================================================
COMMON="data.train_batch_size=$TRAIN_BATCH_SIZE \
actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BATCH_SIZE \
actor_rollout_ref.rollout.n=$ROLLOUT_N \
actor_rollout_ref.model.path=$MODEL \
actor_rollout_ref.actor.policy_loss.loss_mode=sipo \
actor_rollout_ref.actor.optim.lr=$LR \
actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
actor_rollout_ref.actor.self_distillation.alpha=$ALPHA \
actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=True \
actor_rollout_ref.actor.self_distillation.teacher_regularization=$TEACHER \
actor_rollout_ref.actor.self_distillation.teacher_update_rate=$TEACHER_RATE \
actor_rollout_ref.actor.self_distillation.teacher_sync_every=$TEACHER_SYNC \
actor_rollout_ref.actor.ppo_epochs=1 \
algorithm.rollout_correction.rollout_is=token \
data.seed=$SEED"

# beta has to be passed explicitly. COMMON does not set it and actor.yaml defaults it to
# 1.0, which is ten times SDPO's own sweep floor (run_sdpo.sh sweeps BETAS=(0.1 0.5)), so
# leaving it out silently runs a far stronger divergence term than the paper ever used.
# beta scales the analytic divergence LOSS. The RLSD arm has no beta.
ANALYTIC_ARM="actor_rollout_ref.actor.self_distillation.full_logit_distillation=True \
actor_rollout_ref.actor.self_distillation.distillation_topk=100 \
actor_rollout_ref.actor.self_distillation.beta=$BETA \
actor_rollout_ref.actor.self_distillation.reinforce_advantage=False"

case "$MODE" in
    diag)
        # Arm B with lambda pinned at 0: A_d is computed and logged, `advantages` is
        # untouched, so 10 steps of plain GRPO with the teacher's numbers on the log.
        ARM="actor_rollout_ref.actor.self_distillation.full_logit_distillation=False \
actor_rollout_ref.actor.self_distillation.reinforce_advantage=True \
actor_rollout_ref.actor.self_distillation.beta=0.0 \
actor_rollout_ref.actor.self_distillation.sipo_lambda_start=0.0 \
trainer.total_training_steps=10 trainer.test_freq=-1 \
actor_rollout_ref.rollout.val_kwargs.n=4"
        TAG="diag" ;;
    analytic)
        ARM="$ANALYTIC_ARM actor_rollout_ref.rollout.val_kwargs.n=8"
        [ -n "$STEPS" ] && ARM="$ARM trainer.total_training_steps=$STEPS"
        TAG="analytic-a${ALPHA}-b${BETA}" ;;
    reinforce)
        if [ "$ALPHA" != "1.0" ]; then
            echo "ERROR: mode=reinforce is the score-function estimator of REVERSE KL, so it" >&2
            echo "       requires ALPHA=1.0 (got $ALPHA). Use mode=analytic to vary alpha." >&2
            exit 1
        fi
        ARM="actor_rollout_ref.actor.self_distillation.full_logit_distillation=False \
actor_rollout_ref.actor.self_distillation.reinforce_advantage=True \
actor_rollout_ref.actor.self_distillation.beta=0.0 \
actor_rollout_ref.rollout.val_kwargs.n=8"
        [ -n "$STEPS" ] && ARM="$ARM trainer.total_training_steps=$STEPS"
        if [ "$SIPO_FORM" = "additive" ]; then
            TAG="sipoadd-k${SIPO_KAPPA}-e${SIPO_EPS}"
        else
            TAG="sipo-e${SIPO_EPS}-l${SIPO_LAMBDA_START}s${SIPO_LAMBDA_STEPS}"
        fi ;;
esac

DATA_NAME="${DATA_PATH%/}"; DATA_NAME="${DATA_NAME##*/}"
MODEL_NAME="${MODEL##*/}"
EXP_NAME="SIPO-${TAG}-${PRESET}-${PROFILE}-${DATA_NAME}-${MODEL_NAME}-lr${LR}-seed${SEED}"
[[ "$SP" != "1" ]] && EXP_NAME="${EXP_NAME}-sp${SP}"
[[ "$FUSED" == "1" ]] && EXP_NAME="${EXP_NAME}-fused"
[[ "$L_RESP" != "$L_RESP_DEFAULT" ]] && EXP_NAME="${EXP_NAME}-r${L_RESP}"
EXP_NAME="${EXP_NAME}-t${TEACHER}"
[[ "$TEACHER" == "periodic" && "$TEACHER_SYNC" != "10" ]] && EXP_NAME="${EXP_NAME}${TEACHER_SYNC}"
[[ "$TEACHER" == "ema" && "$TEACHER_RATE" != "0.05" ]] && EXP_NAME="${EXP_NAME}${TEACHER_RATE}"
[[ "$SIPO_CTX" == "1" ]] && EXP_NAME="${EXP_NAME}-ctxsipo"
[[ "$MODE" != "analytic" ]] && EXP_NAME="${EXP_NAME}-ctx${SIPO_TEACHER_CTX}"
[[ "$MODE" != "analytic" && "$SIPO_EVIDENCE" != "opd" ]] && EXP_NAME="${EXP_NAME}-ev${SIPO_EVIDENCE}"
[[ "$MODE" != "analytic" && "$SIPO_EVIDENCE" == "contrastive" ]] && EXP_NAME="${EXP_NAME}-neg${CONTRAST_NEG}"
[[ "$MODE" != "analytic" && "$SIPO_EVIDENCE" == "contrastive" && "$CONTRAST_CTX" == "trajectory" ]] && EXP_NAME="${EXP_NAME}-traj"
[[ "$MODE" != "analytic" && "$SIPO_EVIDENCE" == "contrastive" && "$CONTRAST_CTX" == "auto" ]] && EXP_NAME="${EXP_NAME}-auto"
[[ "$MODE" != "analytic" && "$SIPO_EVIDENCE" == "contrastive" && "$CONTRAST_UNJUDGED" == "mode" ]] && EXP_NAME="${EXP_NAME}-unjmode"
[[ "$MODE" != "analytic" && "$SIPO_BASELINE" != "none" ]] && EXP_NAME="${EXP_NAME}-bl${SIPO_BASELINE}"
[[ "$MODE" != "analytic" && "$SIPO_NEG_ONLY" == "1" ]] && EXP_NAME="${EXP_NAME}-negonly"
[[ "$MODE" != "analytic" && "$ALLWRONG_KAPPA" != "0" ]] && EXP_NAME="${EXP_NAME}-aw${ALLWRONG_KAPPA}"
[[ "$IMPUTE" == "1" ]] && EXP_NAME="${EXP_NAME}-impute${IMPUTE_WRONG_SCALE}"
[[ "$IMPUTE" == "1" && "$IMPUTE_CORRECT_SCALE" != "0.0" ]] && EXP_NAME="${EXP_NAME}pos${IMPUTE_CORRECT_SCALE}"
# EXP_SUFFIX: a free-form tag appended to the name, for runs that differ only in settings passed as
# hydra overrides (batch, mini-batch, warmup, ...), which the name does not otherwise carry. It
# also moves the checkpoint directory, so such a run can never resume another run's checkpoint.
[[ -n "${EXP_SUFFIX:-}" ]] && EXP_NAME="${EXP_NAME}-${EXP_SUFFIX}"

# --- resume guard -------------------------------------------------------------
# verl defaults to trainer.resume_mode=auto, and default_local_dir is keyed purely
# on experiment_name (user.yaml:66). So relaunching an identical command silently
# picks up the previous run's global_step_N, sets tqdm initial=N, increments to
# N+1, and immediately evaluates `is_last_step = N+1 >= total_training_steps` as
# True (ray_trainer.py:1426/1740/1743/1784). The run then does exactly one step,
# spends ~50min on the final validation, and exits clean -- which reads like a
# crash after one step. Refuse to guess; make the caller say which they meant.
CKPT_DIR="$PROJECT_ROOT/checkpoints/$EXP_NAME"
TRACKER="$CKPT_DIR/latest_checkpointed_iteration.txt"
RESUME="${RESUME:-}"

# INIT_FROM=<.../checkpoints/<other run>/global_step_N> warm-starts this run from another
# one's weights. Every distillation experiment so far switched the term on at step 0, where
# Dr.GRPO is strongest and the term only competes with it; the regime that motivates the term
# -- reward plateaued, most groups uniform, zero_frac high -- only exists later, and reaching
# it costs a hundred steps of pure RLVR every time. checkpoint.save_contents has no hf_model,
# so the checkpoints are FSDP shards rather than a loadable model directory; symlinking the
# step in and letting verl's own resume pick it up avoids a merge and keeps the optimizer and
# LR schedule, which a fresh Adam at step 100 would throw away.
INIT_FROM="${INIT_FROM:-}"
if [ -n "$INIT_FROM" ]; then
    _step="$(basename "$INIT_FROM")"
    case "$_step" in
        global_step_[0-9]*) _step="${_step#global_step_}" ;;
        *) echo "ERROR: INIT_FROM must point at a global_step_<N> directory, got '$INIT_FROM'" >&2
           exit 1 ;;
    esac
    if [ ! -d "$INIT_FROM" ]; then
        echo "ERROR: INIT_FROM directory does not exist: $INIT_FROM" >&2; exit 1
    fi
    if [ -f "$TRACKER" ]; then
        echo "ERROR: $CKPT_DIR already has checkpoints; INIT_FROM would be ignored." >&2
        echo "       Use RESUME=auto to continue it, or change a knob for a fresh EXP_NAME." >&2
        exit 1
    fi
    mkdir -p "$CKPT_DIR"
    ln -s "$(cd "$(dirname "$INIT_FROM")" && pwd)/global_step_${_step}" "$CKPT_DIR/global_step_${_step}"
    echo "$_step" > "$TRACKER"
    echo "  warm start: $EXP_NAME <- $INIT_FROM (resuming at step $_step)"
    RESUME=auto
    unset _step
fi
RESUME_ARGS=""
if [ -f "$TRACKER" ]; then
    PREV_STEP="$(cat "$TRACKER" 2>/dev/null || echo '?')"
    if [ -z "$RESUME" ]; then
        cat >&2 <<EOF
----------------------------------------------------------------
  已存在 checkpoint: $CKPT_DIR
  上次停在 global_step_${PREV_STEP}

  verl 的 resume_mode 默认是 auto，直接跑会从这里续上。如果
  ${PREV_STEP} >= total_training_steps，它只会走一步就判定训练结束，
  做完 final validation 后退出 —— 看起来很像"跑一步就挂了"。

  请显式选一个：
    RESUME=auto     从 global_step_${PREV_STEP} 续跑（想接着练选这个）
    RESUME=disable  忽略它，从头开始（注意会写进同一个目录）
    SEED=<n> / 换个 BETA 等  换一个 EXP_NAME，开一个干净的目录
----------------------------------------------------------------
EOF
        exit 1
    fi
    RESUME_ARGS="trainer.resume_mode=$RESUME"
elif [ -n "$RESUME" ]; then
    RESUME_ARGS="trainer.resume_mode=$RESUME"
fi
# ------------------------------------------------------------------------------

echo "----------------------------------------------------------------"
echo "  mode / preset / profile : $MODE / $PRESET / $PROFILE"
echo "  experiment              : $EXP_NAME"
echo "  data                    : $DATA_PATH"
echo "  model                   : $MODEL"
echo "  lengths (p/r/reprompt)  : $L_PROMPT / $L_RESP / $L_REPROMPT  (max_model_len=$L_MODEL)"
echo "  fused kernels / SP / TP : $FUSED / $SP / $TP"
echo "  vllm gpu_mem_util       : $GPU_MEM"
echo "  vram / micro_bsz        : ${VRAM_GB}GB per GPU / $MICRO_BSZ"
[[ "$MODE" != "analytic" ]] && echo "  sipo eps / lambda       : $SIPO_EPS / $SIPO_LAMBDA_START -> 0 over $SIPO_LAMBDA_STEPS steps (0 = constant); teacher ctx = $SIPO_TEACHER_CTX; evidence = $SIPO_EVIDENCE (negative = $CONTRAST_NEG, context = $CONTRAST_CTX); baseline = $SIPO_BASELINE; negative-only = $SIPO_NEG_ONLY"
[[ "$IMPUTE" == "1" ]] && echo "  impute                  : all-wrong groups A=-${IMPUTE_WRONG_SCALE}*c, all-correct +${IMPUTE_CORRECT_SCALE}*c; RLSD only on those rows"
[[ "$ALLWRONG_KAPPA" != "0" ]] && echo "  all-wrong groups        : additive contrastive credit A_t = kappa*clip(e_t), kappa = ${ALLWRONG_KAPPA}"
[[ "$MODE" != "analytic" && "$SIPO_FORM" == "additive" ]] && echo "  sipo form               : additive, A_t = A_r + ${SIPO_KAPPA}*clip(e_t, +-${SIPO_EPS}) on every row (lambda unused)"
[[ "$MODE" == "analytic" ]] && echo "  beta / alpha            : $BETA / $ALPHA  (analytic divergence loss)"
if [[ "$TEACHER" == "student" ]]; then
    echo "  teacher                 : student  (unregularized; SDPO Table 4's diverging row)"
elif [[ "$TEACHER" == "periodic" ]]; then
    echo "  teacher                 : periodic snapshot of the student every $TEACHER_SYNC steps (RLSD)"
else
    echo "  teacher                 : $TEACHER, alpha=$TEACHER_RATE  (matches the SIPO repo)"
fi
if [[ "$SIPO_CTX" == "1" ]]; then
    echo "  teacher context         : no incorrect sibling; teacher prompt = privileged block + student prompt verbatim"
else
    echo "  teacher context         : this repo's default (successful rollouts get an incorrect sibling)"
fi
if [ -n "$REWARD_ARGS" ]; then
    echo "  reward                  : acc(0/1) + overlong buffer ${OVERLONG_LEN}/${L_RESP}"
else
    echo "  reward                  : acc(0/1), naive manager (same as gen / rich)"
fi
[ -n "$EVAL_ARGS" ] && echo "  eval                    : math suite (6 benchmarks), val_kwargs.n=${EVAL_N:-1} on pre-repeated rows"
[ ${#EXTRA[@]} -gt 0 ] && echo "  extra overrides         : ${EXTRA[*]}"
[ -n "$RESUME_ARGS" ] && echo "  resume                  : $RESUME (ckpt at global_step_${PREV_STEP:-?})"
if [ "$MODE" != "diag" ]; then
    if [ -n "$STEPS" ]; then
        echo "  steps                   : $STEPS"
    else
        echo "  steps                   : UNCAPPED (total_epochs=30; set STEPS= to cap)"
    fi
fi
echo "----------------------------------------------------------------"

CMD=(bash "$PROJECT_ROOT/training/verl_training.sh" "$EXP_NAME" sipo "$DATA_PATH"
     $PATH_ARGS $MEM_ARGS $COMMON $FEEDBACK_ARGS $REWARD_ARGS $ARM $EVAL_ARGS $RESUME_ARGS
     "trainer.group_name=SIPO-reinforce-adv")
# Appended separately: expanding an empty array inside the literal trips `set -u` on
# bash < 4.4.
if [ ${#CTX_ARGS[@]} -gt 0 ]; then
    CMD+=("${CTX_ARGS[@]}")
fi
if [ ${#EXTRA[@]} -gt 0 ]; then
    CMD+=("${EXTRA[@]}")
fi

LOG="$PROJECT_ROOT/output/${EXP_NAME}.log"

if [ "$DRY_RUN" = true ]; then
    printf '%q ' "${CMD[@]}"; echo
else
    echo "  log                     : $LOG"
    echo "----------------------------------------------------------------"
    # tee, so the opd/* lines survive the terminal. verl's console logger
    # prints one "step:N - key:value - ..." line per step to stdout.
    set -o pipefail
    "${CMD[@]}" 2>&1 | tee "$LOG"
    status=$?
    echo
    exit $status
fi
