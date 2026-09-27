#!/bin/bash
unset VLLM_ATTENTION_BACKEND
export PYTHONBUFFERED=1
# export RAY_DEBUG=1
ulimit -c 0

# Activate conda env for all training runs.
#   VERL_CONDA_ENV=<name>  use a different env
#   VERL_CONDA_ENV=none    skip activation entirely (env already active, venv, container)
CONDA_ENV_NAME="${VERL_CONDA_ENV:-sipo}"
[ "$CONDA_ENV_NAME" = "none" ] && CONDA_ENV_NAME=""

if [ -n "$CONDA_ENV_NAME" ] && [ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV_NAME" ]; then
    # `conda activate` is a shell function, not the conda binary. ~/.bashrc does not
    # define it here: a non-interactive shell returns early from it, which leaves only
    # the binary on PATH and yields "Run 'conda init' before 'conda activate'".
    # Source conda's own hook instead.
    CONDA_BASE="${CONDA_EXE:+$(dirname "$(dirname "$CONDA_EXE")")}"
    [ -n "$CONDA_BASE" ] || CONDA_BASE="$(conda info --base 2>/dev/null)"
    for candidate in "$CONDA_BASE" "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" "/opt/conda"; do
        if [ -n "$candidate" ] && [ -f "$candidate/etc/profile.d/conda.sh" ]; then
            # shellcheck disable=SC1091
            source "$candidate/etc/profile.d/conda.sh"
            break
        fi
    done

    if ! command -v conda >/dev/null 2>&1; then
        echo "ERROR: 'conda' not found. Activate the env yourself and re-run with"
        echo "       VERL_CONDA_ENV=none, or point CONDA_EXE at your conda binary."
        exit 1
    fi

    conda activate "$CONDA_ENV_NAME" || {
        echo "ERROR: Failed to activate conda env '$CONDA_ENV_NAME'."
        echo "       If it is already active, re-run with VERL_CONDA_ENV=none."
        exit 1
    }
fi

export EXPERIMENT=${1:-"experiment"}
CONFIG_NAME=${2:-"ppo_trainer"}
export TASK=${3:-"datasets/ttcs/lasgroup_verifiable-corpus_math-ai_math500_1000"}

# removes the first three arguments from the command line
if [ "$#" -ge 3 ]; then
    shift 3
else
    echo "Usage: $0 <experiment_name> <config_name> <data_path>"
    echo "Example: $0 test ppo_trainer datasets/ttcs/lasgroup_verifiable-corpus_math-ai_math500_1000"
    exit 1
fi

LOSS_MODE=""
for ARG in "$@"; do
    case "$ARG" in
        actor_rollout_ref.actor.policy_loss.loss_mode=*)
            LOSS_MODE="${ARG#actor_rollout_ref.actor.policy_loss.loss_mode=}"
            ;;
    esac
done

# Keep config name aligned with explicit loss mode when both are sdpo/sipo.
if [[ "$CONFIG_NAME" =~ ^(sdpo|sipo)$ ]] && [[ "$LOSS_MODE" =~ ^(sdpo|sipo)$ ]] && [ "$CONFIG_NAME" != "$LOSS_MODE" ]; then
    echo "INFO: config_name '$CONFIG_NAME' conflicts with loss_mode '$LOSS_MODE'; using '$LOSS_MODE'."
    CONFIG_NAME="$LOSS_MODE"
fi

echo "Experiment: $EXPERIMENT"
echo "Config file: $CONFIG_NAME"
if [ -n "$LOSS_MODE" ]; then
    echo "Policy loss mode: $LOSS_MODE"
fi
echo "Task: $TASK"
echo "Arguments: $@"

python -m verl.trainer.main_ppo --config-name "$CONFIG_NAME" "$@"
