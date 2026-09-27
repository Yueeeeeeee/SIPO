#!/bin/bash
# Install flash-attn for the torch already in the env.
#
# DEFAULT: a PREBUILT wheel. Nothing is compiled, so `nvcc` is never invoked and the local
# CUDA toolkit is irrelevant. This is the path that works on a box whose toolkit is newer
# than torch's CUDA.
#
# Why compiling is the fallback and not the default: torch itself refuses the build. When
# a CUDAExtension has .cu sources, BuildExtension.build_extensions() calls
# _check_cuda_version(), which compares the CUDA MAJOR of the local nvcc against
# torch.version.cuda and raises unconditionally when they differ:
#   RuntimeError: The detected CUDA version (13.4) mismatches the version that was
#                 used to compile PyTorch (12.8)
# That check lives in torch, not in flash-attn, so patching flash-attn's setup.py -- which
# is what the zipzou/flash-attention fork does -- does not get past it. Only a matching
# CUDA major does. TORCH_DONT_CHECK_COMPILER_ABI covers the compiler ABI check, not this one.
#
# The wheel must match four things: python tag, torch MAJOR.MINOR, CUDA major, cxx11 ABI.
# Coverage is thin -- as of writing, torch2.9 + cu12 + abiTRUE wheels exist only for cp312 --
# so a Python other than 3.12 on torch 2.9 has no wheel and needs --build.
#
# Newest is not always right. flash-attn 2.8.3 added the flash_attn/cute (CuTe DSL / FA4)
# backend, which needs a matching nvidia-cutlass-dsl; with a mismatched one, vLLM's rotary
# embedding import chain reaches it and dies with
#   AttributeError: module 'cutlass.cute.core' has no attribute 'ThrMma'
# 2.8.1 has no cute subpackage at all (deps: torch, einops) and still carries everything
# verl uses: bert_padding, ops/triton, flash_attn_interface. Hence WANT_VERSION.
#
# Usage:
#   bash scripts/install_flash_attn.sh              # prebuilt wheel, recommended version
#   bash scripts/install_flash_attn.sh --list       # list candidate wheels, install nothing
#   bash scripts/install_flash_attn.sh 2.8.3        # a specific version
#   bash scripts/install_flash_attn.sh --newest     # newest matching wheel
#   bash scripts/install_flash_attn.sh --build      # compile from source (see above)
set -euo pipefail

# Known-good default: no cute/cutlass dependency, has every module verl imports.
WANT_VERSION="2.8.1"

LIST_ONLY=false
PICK_NEWEST=false
DO_BUILD=false
case "${1:-}" in
    --list)   LIST_ONLY=true ;;
    --newest) PICK_NEWEST=true ;;
    --build)  DO_BUILD=true ;;
    "")       ;;
    *)        WANT_VERSION="$1" ;;
esac

eval "$(python - <<'PY'
import sys, torch
v = torch.__version__.split('+')[0].split('.')
cu = torch.version.cuda or ''
print(f'PY_TAG=cp{sys.version_info.major}{sys.version_info.minor}')
print(f'TORCH_MM={v[0]}.{v[1]}')
print(f'TORCH_FULL={torch.__version__}')
print(f'CU_TAG=cu{cu.split(".")[0] if cu else "NONE"}')
print(f'ABI_FLAG={"TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE"}')
PY
)"

echo "torch      : $TORCH_FULL  (tag torch$TORCH_MM, $CU_TAG)"
echo "python     : $PY_TAG"
echo "cxx11 ABI  : $ABI_FLAG"
echo "nvcc       : $(command -v nvcc >/dev/null 2>&1 && nvcc --version | grep -o 'release [0-9.]*' || echo 'not on PATH')"
echo

# ---- source build --------------------------------------------------------------------
if [ "$DO_BUILD" = true ]; then
    NVCC_MAJOR="$(command -v nvcc >/dev/null 2>&1 \
        && nvcc --version | grep -o 'release [0-9]*' | cut -d' ' -f2 || true)"
    TORCH_CU_MAJOR="${CU_TAG#cu}"
    if [ -z "$NVCC_MAJOR" ]; then
        echo "ERROR: --build needs the CUDA toolkit; nvcc is not on PATH." >&2
        echo "       export PATH=/usr/local/cuda/bin:\$PATH" >&2
        exit 1
    fi
    if [ "$NVCC_MAJOR" != "$TORCH_CU_MAJOR" ]; then
        echo "ERROR: nvcc is CUDA $NVCC_MAJOR.x but torch was built against CUDA $TORCH_CU_MAJOR.x." >&2
        echo "       torch's own cpp_extension._check_cuda_version() raises on a major mismatch" >&2
        echo "       during build_extensions(), so no flash-attn fork can work around it." >&2
        echo "       Either drop --build and take a prebuilt wheel, or put a matching toolkit" >&2
        echo "       in this env without touching the system one:" >&2
        echo "         conda install -c nvidia cuda-toolkit=${TORCH_CU_MAJOR}.8" >&2
        echo "         export CUDA_HOME=\$CONDA_PREFIX PATH=\$CONDA_PREFIX/bin:\$PATH" >&2
        exit 1
    fi
    MISSING=""
    for mod in packaging ninja setuptools wheel; do
        python -c "import $mod" >/dev/null 2>&1 || MISSING="$MISSING $mod"
    done
    # --no-build-isolation is required (the build imports torch to read its ABI) and means
    # pip installs no build dependencies of its own.
    [ -z "$MISSING" ] || { echo "ERROR: --no-build-isolation installs no build deps; missing:$MISSING" >&2
                           echo "       pip install$MISSING" >&2; exit 1; }
    if [ -z "${MAX_JOBS:-}" ]; then
        CORES="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)"
        MAX_JOBS=$(( CORES / 4 )); [ "$MAX_JOBS" -ge 1 ] || MAX_JOBS=1
    fi
    export MAX_JOBS
    REPO="${FLASH_ATTN_REPO:-https://github.com/zipzou/flash-attention.git}"
    REF="${FLASH_ATTN_REF:-main}"
    echo "Building $REPO @ $REF with MAX_JOBS=$MAX_JOBS (30-90 min; a bare 'Killed' is OOM)."
    pip install --no-build-isolation --no-cache-dir "git+${REPO}@${REF}"
    python -c "import flash_attn; from flash_attn.bert_padding import pad_input; print('flash_attn', flash_attn.__version__, 'OK')"
    exit 0
fi


API="https://api.github.com/repos/Dao-AILab/flash-attention/releases?per_page=100"
ASSETS="$(curl -fsSL "$API" | grep -o 'https://github.com/Dao-AILab/flash-attention/releases/download/[^"]*\.whl' | sort -u)"
[ -n "$ASSETS" ] || { echo "ERROR: could not list release assets (network/rate limit)."; exit 1; }

WANT="${CU_TAG}torch${TORCH_MM}cxx11abi${ABI_FLAG}-${PY_TAG}-${PY_TAG}-linux_x86_64.whl"
MATCHES="$(printf '%s\n' "$ASSETS" | grep -F "$WANT" | sort -Vr || true)"

if [ -z "$MATCHES" ]; then
    echo "No wheel matches $WANT"
    echo
    echo "torch tags that DO have ${PY_TAG} / abi${ABI_FLAG} wheels:"
    printf '%s\n' "$ASSETS" | grep -F "cxx11abi${ABI_FLAG}-${PY_TAG}-" \
        | grep -oE "${CU_TAG}torch[0-9]+\.[0-9]+" | sort -Vu | sed 's/^/  /'
    echo
    echo "Three ways out, cheapest first:"
    echo "  1. Match a python that has wheels for this torch. torch2.9 + cu12 + abiTRUE"
    echo "     currently ships cp312 only, so Python 3.12 is what this stack expects."
    echo "  2. Pick a torch whose tag is listed above (wheels are usually ABI-compatible one"
    echo "     minor up -- Dao-AILab/flash-attention#1644) and reinstall vllm to match:"
    echo "     vllm 0.10.0 -> torch 2.7.1, 0.11.x -> 2.8.0, 0.12.0 -> 2.9.x."
    echo "  3. Compile: bash scripts/install_flash_attn.sh --build. Needs a CUDA toolkit"
    echo "     whose MAJOR matches torch's ($CU_TAG), because torch refuses otherwise."
    exit 1
fi

echo "Matching wheels (newest first):"
printf '%s\n' "$MATCHES" | sed 's|.*/||' | sed 's/^/  /'
echo

if [ "$LIST_ONLY" = true ]; then exit 0; fi

if [ "$PICK_NEWEST" = true ]; then
    URL="$(printf '%s\n' "$MATCHES" | head -1)"
else
    URL="$(printf '%s\n' "$MATCHES" | grep -F "flash_attn-${WANT_VERSION}+" | head -1 || true)"
    if [ -z "$URL" ]; then
        echo "No wheel for flash-attn ${WANT_VERSION} with this torch/python/ABI."
        echo "Pass a version listed above, or --newest to take the top one."
        exit 1
    fi
fi
echo "Installing $(basename "$URL")"
pip install --no-cache-dir "$URL"
python -c "import flash_attn; from flash_attn.bert_padding import pad_input; print('flash_attn', flash_attn.__version__, 'OK')"
