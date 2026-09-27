## Getting Started

### System Requirements

*   **OS:** Linux (Ubuntu 22.04)
*   **Python:** 3.12
*   **GPUs:** a single node of 8. Verified on 8x A100 40GB and 8x H100 80GB.
    The runner reads the card size from `nvidia-smi` and widens the micro batch on 80GB
    cards -- see [Training](#training).
*   **CUDA driver:** anything that runs the cu12.8 wheels below.
*   **CUDA toolkit:** not needed. Every CUDA-dependent package is a prebuilt wheel, so
    `nvcc` is never invoked and a box with CUDA 13.4 runs this cu12.8 stack fine. Only
    `scripts/install_flash_attn.sh --build` needs one, and then its major version has to
    match torch's.

---

### Installation

The documented training commands target the following stack. Run from the repository root
and install in this order -- vLLM pulls its own torch, so
anything compiled against torch must come after it.

| component | version | notes |
|---|---|---|
| Python | 3.12 | conda env; `cp312` wheel tag |
| vLLM | **0.12.0** | pulls torch 2.9.x + cu12.8 |
| torch | **2.9.x + cu12.8** | installed by vLLM, do not pin separately |
| flash-attn | **2.8.1** | prebuilt wheel, `cu12torch2.9cxx11abiTRUE-cp312`; **not** 2.8.3 |
| ray | 2.53.0 | from `requirements.txt` |
| opentelemetry | api/sdk/proto 1.39.1, semconv/exporter-prometheus 0.60b1 | pinned in `requirements.txt` |

```bash
conda create -n sipo python=3.12 -y && conda activate sipo
pip install -r requirements.txt
pip install -e .

pip install vllm==0.12.0            # brings torch 2.9.x + cu12.8
bash scripts/install_flash_attn.sh  # prebuilt flash-attn wheel for that torch

# vLLM can move the opentelemetry pins; restore them if it did
pip install "opentelemetry-api==1.39.1" "opentelemetry-sdk==1.39.1" \
            "opentelemetry-proto==1.39.1" \
            "opentelemetry-semantic-conventions==0.60b1" \
            "opentelemetry-exporter-prometheus==0.60b1"
```

Verify all four before running anything -- each one maps to a failure that otherwise
surfaces far from its cause:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
python -c "from vllm.v1.engine.utils import CoreEngineProcManager; print('vllm OK')"
python -c "from flash_attn.bert_padding import pad_input, unpad_input; print('flash-attn OK')"
python -c "from opentelemetry.exporter.prometheus import PrometheusMetricReader; print('otel OK')"
```

### Rollout Engine Compatibility

`verl/trainer/config/user.yaml` sets `rollout.name: vllm`, and vLLM is **not** in
`requirements.txt` -- install it separately. The version is constrained by the async rollout
server, which imports `vllm.v1.engine.utils.CoreEngineProcManager` unconditionally
(absent in 0.8.x/0.9.x) and branches explicitly on 0.11.0 and 0.12.0
(`verl/workers/rollout/vllm_rollout/vllm_async_server.py:67-79`).

| torch | vLLM | evidence |
|---|---|---|
| 2.9.x + cu128 | **0.12.0** | Stack used by this guide; dedicated code branches |
| 2.8.0 + cu128 | 0.11.x | Historical upstream Docker configuration; not verified here |
| 2.7.1 + cu126 | 0.10.0 | Historical upstream Docker configuration; not verified here |

flash-attn must be reinstalled whenever torch changes -- it is compiled against torch's
ABI and CUDA version, so a torch upgrade leaves a binary that no longer loads.

`scripts/install_flash_attn.sh` installs a **prebuilt wheel**. Nothing is compiled, which
is the whole point: `pip install flash-attn` builds from source, and a source build is
refused whenever the local CUDA toolkit's major version differs from torch's.

> [!IMPORTANT]
> That refusal comes from **torch**, not from flash-attn. When a CUDAExtension has `.cu`
> sources, `BuildExtension.build_extensions()` calls
> `torch/utils/cpp_extension.py::_check_cuda_version()`, which compares the CUDA *major* of
> the local `nvcc` against `torch.version.cuda` and raises unconditionally when they differ:
>
> ```
> RuntimeError: The detected CUDA version (13.4) mismatches the version that was
>               used to compile PyTorch (12.8). Please make sure to use the same CUDA versions.
> ```
>
> Because the check is torch's, patching flash-attn's own `setup.py` does not get past it --
> forks that relax the parity check (e.g. `zipzou/flash-attention`) still hit this.
> `TORCH_DONT_CHECK_COMPILER_ABI` covers the compiler ABI check, not this one. Only a
> matching CUDA major, or not compiling at all, works.

The wheel has to match four things: **python tag, torch MAJOR.MINOR, CUDA major, cxx11
ABI**. Coverage is thinner than it looks -- as of writing, `torch2.9 + cu12 + abiTRUE`
ships **cp312 only** -- which is why this stack pins Python 3.12. Run `--list` to see the
candidates for the current environment rather than guessing; `--newest` takes the top one
and a bare version argument picks that one.

It defaults to **2.8.1**, not the newest: 2.8.3 added the `flash_attn/cute` (CuTe DSL)
backend, whose `nvidia-cutlass-dsl` requirement vLLM's rotary-embedding import chain walks
into, failing with `module 'cutlass.cute.core' has no attribute 'ThrMma'`. 2.8.1 has no
`cute` subpackage (deps: `torch`, `einops`) and still provides everything verl imports:
`bert_padding`, `ops/triton`, `flash_attn_interface`.

The failure surfaces at model load, not at install, so a 2.8.3 wheel installed by hand
(picking the newest off `--list`, say) looks fine until the first run. To recover:

```bash
pip uninstall -y flash-attn
bash scripts/install_flash_attn.sh            # 2.8.1

python -c "import flash_attn; print(flash_attn.__version__)"   # 2.8.1
python -c "import flash_attn.cute" 2>&1 | tail -1              # want ModuleNotFoundError
```

A `ModuleNotFoundError` on the second line is the point: it confirms the subpackage that
triggers the cutlass import is gone. Keeping 2.8.3 and installing a matching
`nvidia-cutlass-dsl` also works, but the pairing is undocumented and 2.8.3 adds nothing
verl uses.

**If no wheel matches**, the script prints the three ways out. Compiling is the last of
them (`--build`, source from `zipzou/flash-attention` with `--no-build-isolation`), and it
needs a CUDA toolkit whose major matches torch's. One can be put in the conda env without
touching the system toolkit:

```bash
conda install -c nvidia cuda-toolkit=12.8
export CUDA_HOME=$CONDA_PREFIX PATH=$CONDA_PREFIX/bin:$PATH
bash scripts/install_flash_attn.sh --build
```

flash-attn is not optional on CUDA: `verl/utils/attention_utils.py` imports
`flash_attn.bert_padding` unconditionally, and `use_remove_padding=True` (the default) goes
through it.

For SGLang instead of vLLM, see `SGLANG_REQUIRES` in `setup.py`
(`sglang[srt,openai]==0.5.6`, `torch==2.9.1`) and pass `ROLLOUT=sglang` to the runner, but
note the warning about `rollout_log_probs` in that script.

The dependency lists in `setup.py` are currently inactive: both `install_requires` and
`extras_require` are commented out in `setup()`. Consequently, `pip install -e .` does
not install those dependencies or enforce the listed vLLM range. Likewise,
`pip install -e ".[sglang]"` does not install SGLang. Install engine dependencies explicitly.

### Requirement Files

| File | Description |
|------|-------------|
| `requirements.txt` | Core dependencies, mostly pinned. Does **not** include vLLM or flash-attn -- both are installed separately above, because both must match the torch that vLLM brings. |

> [!NOTE]
> For verl architecture and advanced configuration beyond what is documented here, see the
> [official verl repository](https://github.com/volcengine/verl).

---

### Data Preparation

The data is loaded from `train.parquet` and `test.parquet` (see `verl/trainer/config/user.yaml`).
Each dataset directory should contain `train.json` and `test.json`, then be preprocessed into parquet.

Run the data scripts from the repository root (they are invoked as `python -m data.<script>`,
which puts the repository on `sys.path`).

No math data is checked in. On a fresh machine you need two things before you can train on
math: the **training set**, and the **evaluation suite** that validation reads.

#### Math: the full setup, start to finish

Run all of this from the repository root, with the `sipo` env active.

**1. Training set -- DAPO-Math-17k.**

```bash
python -m data.load_dataset --dataset_name dapo_math --category en \
  --output_path datasets/dapo_math.json
python -m data.split_tasks --json_path datasets/dapo_math.json \
  --output_dir datasets/dapo_math --test_ratio 0.05 --seed 42
python -m data.preprocess --data_source datasets/dapo_math
```

`--category` is `all` (17398 rows) / `en` (14116) / `cn` (3282); `en` is what the commands
below assume. The source is `open-r1/DAPO-Math-17k-Processed`, the deduplicated release --
the original `BytedTsinghua-SIA/DAPO-Math-17k` parquet is pre-expanded to 1.79M rows for
DAPO's own rollout scheme and is not a usable prompt set.

`split_tasks` writes `train.json` / `test.json`; `preprocess` turns them into the
`train.parquet` / `test.parquet` that the trainer actually reads. All three steps are
required -- the trainer never reads JSON.

**2. Evaluation suite -- AIME24, AIME25, AMC23, MATH-500, Minerva Math, OlympiadBench.**

```bash
bash data/build_math_eval_suite.sh
```

Writes `datasets/math_eval/<name>/test.parquet`, one parquet per benchmark. verl keys
validation metrics on `data_source`, so each reports separately as
`val-core/<name>/acc/mean@1`. Pass `EVAL_SUITE=1` to the runner to use them instead of the
training set's own test split.

`avg@k` is baked into the parquets by duplicating rows, and validation then runs at
`val_kwargs.n=1` -- verl applies `val_kwargs.n` uniformly to every val file, so duplication
is the only way to sample a 30-problem set 32 times without also sampling a 674-problem set
32 times:

| benchmark | problems | repeat | generations |
|---|---|---|---|
| aime24 / aime25 | 30 / 30 | 32 | 960 / 960 |
| amc23 | 40 | 8 | 320 |
| math500 | 500 | 1 | 500 |
| minervamath | 272 | 1 | 272 |
| olympiadbench | 674 | 1 | 674 |
| | | | **3686** |

Override a factor with `REPEAT_AIME24=...` etc. One validation pass costs roughly 50-90
minutes at `max_response_length=12288`, so budget for it when choosing `trainer.test_freq`.

**3. Check before launching.**

```bash
ls datasets/dapo_math/{train,test}.parquet
ls datasets/math_eval/*/test.parquet          # expect exactly 6
```

The shell runner does not check that these files exist. The trainer loads the validation
dataset during initialization, even with `trainer.val_before_train=False`, so a missing
eval parquet prevents training from starting.

#### Data Loading
The detailed instructions for loading data are in `data/README.md`.

To regenerate all SciKnowEval domains (Biology/Chemistry/Material/Physics):
```bash
for spec in "Biology biology" "Chemistry chemistry" "Material material" "Physics physics"; do
  NAME="${spec%% *}"
  DIR="${spec#* }"

  python -m data.load_dataset \
    --dataset_name "$NAME" \
    --output_path "datasets/sciknoweval/$DIR/$DIR.json"

  python -m data.split_tasks \
    --json_path "datasets/sciknoweval/$DIR/$DIR.json" \
    --output_dir "datasets/sciknoweval/$DIR" \
    --test_ratio 0.1 \
    --seed 42
done

```

To create `datasets/lcb_v6/train.json` and `datasets/lcb_v6/test.json`:
```bash
python -m data.load_dataset \
  --dataset_name livecodebench/code_generation_lite-v6 \
  --output_path datasets/lcb_v6.json

python -m data.split_tests \
  --json_path datasets/lcb_v6.json \
  --output_dir datasets/lcb_v6
python -m data.preprocess --data_source datasets/lcb_v6
```


#### Data Preprocessing
Our implementation uses the `parquet` format for the data. To preprocess the data, run the following command:

```bash
python -m data.preprocess --data_source DATASET_PATH
```
`DATASET_PATH` should contain the `train.json` and `test.json` files, and the `train.parquet`
and `test.parquet` files are written back into that same directory.

**This is required once per dataset directory** -- no parquet files are checked in, and the
trainer only ever reads parquet (see `data.train_files` / `data.val_files` in
`verl/trainer/config/user.yaml`). The step is not a plain format conversion: it builds the
verl chat-format prompt, applies the SciKnowEval multiple-choice instruction, appends the
student reasoning suffix, swaps in `tests` as the ground truth for code tasks, and casts to
`LargeString`/`LargeList` to avoid Arrow 32-bit offset overflow.

The five dataset directories that ship with `train.json` / `test.json`:
```bash
for d in datasets/tooluse \
         datasets/sciknoweval/biology datasets/sciknoweval/chemistry \
         datasets/sciknoweval/material datasets/sciknoweval/physics; do
  python -m data.preprocess --data_source "$d"
done
```
`datasets/lcb_v6` has no JSON checked in, so run the LiveCodeBench download and split above
before preprocessing it.

If you are on NFS and see temporary-file cleanup issues from multiprocessing, run preprocessing in single-process mode:
```bash
python - <<'PY'
from data.preprocess import run_proprocessing
for d in [
    "datasets/sciknoweval/biology",
    "datasets/sciknoweval/chemistry",
    "datasets/sciknoweval/material",
    "datasets/sciknoweval/physics",
    "datasets/tooluse",
    "datasets/lcb_v6",
]:
    run_proprocessing(d, num_proc=1)
PY
```

To inspect the first example in a parquet file:
```bash
python - <<'PY'
import datasets
ds = datasets.load_dataset("parquet", data_files="datasets/sciknoweval/biology/train.parquet", split="train")
print(ds[0])
PY
```

---

### Configuration
The training commands below automatically set repository, log and checkpoint paths to
the repository root, `output/` and `checkpoints/`, respectively. No edits to
`verl/trainer/config/user.yaml` are required for those commands.

If you invoke `training/verl_training.sh` directly, adapt these paths in
`verl/trainer/config/user.yaml` or supply equivalent command-line overrides:

```yaml
vars:
  dir: /path/to/your/SIPO              # Path to this repository
  log_dir: /path/to/your/logs          # Directory for logs
  ckpt_dir: /path/to/your/checkpoints  # Directory for model checkpoints
```

`data.train_files`, `data.val_files` and `custom_reward_function.path` derive from
`vars.dir`. The training wrapper activates the `sipo` conda environment by default; use
`VERL_CONDA_ENV=<name>` for another environment or `VERL_CONDA_ENV=none` to use an already
activated environment without conda activation.

Training logs to Weights & Biases by default (`trainer.logger: ["console", "wandb"]`), so
either `wandb login` first or append `'trainer.logger=[console]'` to the training command
(the quotes also prevent zsh from treating the brackets as a filename pattern).

---

### Training

One section per task. Every command is written out in full, with no shell
variables, so it pastes the same into bash and zsh. `reinforce` is the script's first
positional argument (a `MODE` environment variable is ignored); append `--dry-run` to print
the composed command and the experiment name without launching. The offload overrides assume
80GB cards.

Math runs 128 prompts per step with one optimizer update per step. Science, tool use and code
follow the original SDPO runs: 32 prompts x 8 rollouts per step and micro batch 1; science and
tool use with mini-batch 32, lr 1e-5 and 10 warmup steps, code with mini-batch 8 and lr 1e-6.
Each of those three sections gives our method first, then GRPO under the same settings (the
teacher term switched off).

#### Math (DAPO-Math-17k)

Additive contrastive credit at kappa = 0.5; a wrong rollout without
a judged answer of its own gets no teacher term.

```bash
SIPO_FORM=additive SIPO_KAPPA=0.5 CONTRAST_NEG=own \
SIPO_EVIDENCE=contrastive SIPO_EPS=0.2 SIPO_TEACHER_CTX=gold \
SIPO_CTX=1 RESP=12288 PRESET=math STEPS=125 PROFILE=full FUSED=1 EVAL_SUITE=1 \
bash experiments/reinforce_adv/run_sipo_reinforce_adv.sh reinforce \
  data.train_batch_size=128 actor_rollout_ref.actor.ppo_mini_batch_size=128 \
  trainer.test_freq=25 trainer.save_freq=25 trainer.val_before_train=False
```

#### Science (SciKnowEval)

One run per subject: after `chemistry`, run `biology`, `material` and `physics`.

```bash
# ours: sibling trajectory pair where both siblings exist, else gold vs the row's own wrong letter
MICRO_BSZ=1 GPU_MEM=0.55 EXP_SUFFIX=sdpo LR=1e-5 \
PRESET=gen DATA_PATH=datasets/sciknoweval/chemistry \
SIPO_FORM=additive SIPO_KAPPA=0.15 CONTRAST_NEG=own CONTRAST_CTX=auto \
SIPO_EVIDENCE=contrastive SIPO_EPS=0.2 SIPO_TEACHER_CTX=gold \
SIPO_CTX=1 STEPS=125 PROFILE=full FUSED=1 \
bash experiments/reinforce_adv/run_sipo_reinforce_adv.sh reinforce \
  trainer.test_freq=5 trainer.save_freq=25 trainer.val_before_train=False \
  actor_rollout_ref.rollout.val_kwargs.n=16 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False

# GRPO
MICRO_BSZ=1 GPU_MEM=0.55 EXP_SUFFIX=sdpo LR=1e-5 \
PRESET=gen DATA_PATH=datasets/sciknoweval/chemistry \
SIPO_LAMBDA_START=0 SIPO_LAMBDA_STEPS=0 SIPO_EPS=0.2 SIPO_TEACHER_CTX=gold \
SIPO_CTX=1 STEPS=125 PROFILE=full FUSED=1 \
bash experiments/reinforce_adv/run_sipo_reinforce_adv.sh reinforce \
  trainer.test_freq=5 trainer.save_freq=25 trainer.val_before_train=False \
  actor_rollout_ref.rollout.val_kwargs.n=16 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
```

#### Tool use

```bash
# ours: gold action list, rendered in the scorer's format, vs the row's own wrong one
MICRO_BSZ=1 GPU_MEM=0.55 EXP_SUFFIX=sdpo LR=1e-5 \
PRESET=gen DATA_PATH=datasets/tooluse \
SIPO_FORM=additive SIPO_KAPPA=0.15 CONTRAST_NEG=own CONTRAST_CTX=answer \
SIPO_EVIDENCE=contrastive SIPO_EPS=0.2 SIPO_TEACHER_CTX=gold \
SIPO_CTX=1 STEPS=125 PROFILE=full FUSED=1 \
bash experiments/reinforce_adv/run_sipo_reinforce_adv.sh reinforce \
  trainer.test_freq=5 trainer.save_freq=25 trainer.val_before_train=False \
  actor_rollout_ref.rollout.val_kwargs.n=16 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False

# GRPO
MICRO_BSZ=1 GPU_MEM=0.55 EXP_SUFFIX=sdpo LR=1e-5 \
PRESET=gen DATA_PATH=datasets/tooluse \
SIPO_LAMBDA_START=0 SIPO_LAMBDA_STEPS=0 SIPO_EPS=0.2 SIPO_TEACHER_CTX=gold \
SIPO_CTX=1 STEPS=125 PROFILE=full FUSED=1 \
bash experiments/reinforce_adv/run_sipo_reinforce_adv.sh reinforce \
  trainer.test_freq=5 trainer.save_freq=25 trainer.val_before_train=False \
  actor_rollout_ref.rollout.val_kwargs.n=16 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
```

#### Code (LCBv6)

```bash
# ours: passing vs failing sibling code; all-wrong groups get no contrast
MICRO_BSZ=1 GPU_MEM=0.55 EXP_SUFFIX=sdpo LR=1e-6 \
PRESET=rich \
SIPO_FORM=additive SIPO_KAPPA=0.15 CONTRAST_CTX=trajectory \
SIPO_EVIDENCE=contrastive SIPO_EPS=0.2 SIPO_TEACHER_CTX=sipo \
SIPO_CTX=1 STEPS=125 PROFILE=full FUSED=1 \
bash experiments/reinforce_adv/run_sipo_reinforce_adv.sh reinforce \
  trainer.test_freq=5 trainer.save_freq=25 trainer.val_before_train=False \
  actor_rollout_ref.rollout.val_kwargs.n=4 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
  actor_rollout_ref.actor.self_distillation.environment_feedback_only_without_solution=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False

# GRPO
MICRO_BSZ=1 GPU_MEM=0.55 EXP_SUFFIX=sdpo LR=1e-6 \
PRESET=rich \
SIPO_LAMBDA_START=0 SIPO_LAMBDA_STEPS=0 SIPO_EPS=0.2 SIPO_TEACHER_CTX=sipo \
SIPO_CTX=1 STEPS=125 PROFILE=full FUSED=1 \
bash experiments/reinforce_adv/run_sipo_reinforce_adv.sh reinforce \
  trainer.test_freq=5 trainer.save_freq=25 trainer.val_before_train=False \
  actor_rollout_ref.rollout.val_kwargs.n=4 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
  actor_rollout_ref.actor.self_distillation.environment_feedback_only_without_solution=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
```
