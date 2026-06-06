#!/usr/bin/env bash
# =============================================================================
# FlatQuant one-shot environment setup.
#
# Usage (from a fresh clone, inside an activated conda env):
#     conda create -n flatquant python=3.10 -y && conda activate flatquant
#     bash setup.sh
#
# What it does:
#   1. Installs Python requirements.
#   2. Makes sure a CUDA toolkit (nvcc) matching your torch build is available
#      (installs one into the conda env via conda if nvcc is missing).
#   3. Builds the `fast_hadamard_transform` CUDA extension (a core dependency),
#      patching it for CUDA 13 (which dropped the compute_70 / Volta target).
#   4. Installs the FlatQuant package itself (core only, skipping the optional
#      `deploy._CUDA` inference kernels which need cmake + the cutlass submodule).
#
# To ALSO build the deploy/inference kernels, set BUILD_DEPLOY_KERNELS=1.
# =============================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "ERROR: no active conda env detected (CONDA_PREFIX is empty)."
    echo "       Run:  conda create -n flatquant python=3.10 -y && conda activate flatquant"
    exit 1
fi

PY="$(command -v python)"
PIP="$PY -m pip"
echo ">>> Using python: $PY"

# ---------------------------------------------------------------------------
# 1. Python requirements
# ---------------------------------------------------------------------------
echo ">>> [1/4] Installing Python requirements..."
$PIP install -r requirements.txt
$PIP install ninja                      # ninja => fast extension builds

# ---------------------------------------------------------------------------
# 2. CUDA toolkit (nvcc) matching the installed torch
# ---------------------------------------------------------------------------
echo ">>> [2/4] Checking CUDA toolchain..."
TORCH_CUDA="$($PY -c 'import torch; print(torch.version.cuda or "")' 2>/dev/null || true)"
if [[ -z "$TORCH_CUDA" ]]; then
    echo "ERROR: torch reports no CUDA build. Install a CUDA build of torch first."
    exit 1
fi
echo "    torch CUDA version: $TORCH_CUDA"

export CUDA_HOME="${CUDA_HOME:-$CONDA_PREFIX}"
if ! command -v nvcc >/dev/null 2>&1 && [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
    echo "    nvcc not found -> installing cuda-toolkit=$TORCH_CUDA into the conda env..."
    if command -v conda >/dev/null 2>&1; then
        conda install -y -p "$CONDA_PREFIX" -c nvidia "cuda-toolkit=${TORCH_CUDA}"
    else
        echo "ERROR: nvcc missing and 'conda' not on PATH. Install a CUDA toolkit matching $TORCH_CUDA."
        exit 1
    fi
fi
export PATH="$CUDA_HOME/bin:$PATH"
# conda's cuda-toolkit puts headers/libs under targets/<arch>/ rather than include/lib
TGT_INC="$CUDA_HOME/targets/x86_64-linux/include"
TGT_LIB="$CUDA_HOME/targets/x86_64-linux/lib"
[[ -d "$TGT_INC" ]] && export CPATH="${TGT_INC}${CPATH:+:$CPATH}"
[[ -d "$TGT_LIB" ]] && export LIBRARY_PATH="${TGT_LIB}${LIBRARY_PATH:+:$LIBRARY_PATH}"
echo "    nvcc: $(nvcc --version | sed -n 's/.*release /release /p')"

# ---------------------------------------------------------------------------
# 3. fast_hadamard_transform (core CUDA dependency)
# ---------------------------------------------------------------------------
echo ">>> [3/4] Building fast_hadamard_transform..."
# Fetch the kernel sources. In the upstream repo these live in a git submodule;
# in a vendored copy (e.g. this repo) the sources are already present.
if [[ ! -f third-party/fast-hadamard-transform/setup.py ]]; then
    git submodule update --init third-party/fast-hadamard-transform
fi
FHT_SETUP="third-party/fast-hadamard-transform/setup.py"
# Patch: CUDA >= 13 dropped compute_70; ensure modern Ampere+ targets are present.
$PY - "$FHT_SETUP" <<'PYEOF'
import re, sys
p = sys.argv[1]
s = open(p).read()
if "compute_86" not in s:
    s = s.replace(
        '    cc_flag.append("-gencode")\n'
        '    cc_flag.append("arch=compute_70,code=sm_70")\n'
        '    cc_flag.append("-gencode")\n'
        '    cc_flag.append("arch=compute_80,code=sm_80")\n',
        '    if bare_metal_version < Version("13.0"):\n'
        '        cc_flag.append("-gencode")\n'
        '        cc_flag.append("arch=compute_70,code=sm_70")\n'
        '    cc_flag.append("-gencode")\n'
        '    cc_flag.append("arch=compute_80,code=sm_80")\n'
        '    cc_flag.append("-gencode")\n'
        '    cc_flag.append("arch=compute_86,code=sm_86")\n',
    )
    open(p, "w").write(s)
    print("    patched fht setup.py for CUDA 13 / sm_86")
else:
    print("    fht setup.py already patched")
PYEOF
$PIP install -e third-party/fast-hadamard-transform --no-build-isolation

# ---------------------------------------------------------------------------
# 4. FlatQuant package
# ---------------------------------------------------------------------------
echo ">>> [4/4] Installing FlatQuant..."
if [[ "${BUILD_DEPLOY_KERNELS:-0}" == "1" ]]; then
    echo "    building WITH deploy._CUDA kernels (needs cmake + cutlass)..."
    $PIP install cmake
    if [[ ! -d third-party/cutlass/include ]]; then
        git submodule update --init third-party/cutlass 2>/dev/null \
            || git clone --depth 1 https://github.com/NVIDIA/cutlass.git third-party/cutlass
    fi
    $PIP install -e . --no-build-isolation
else
    FLATQUANT_SKIP_DEPLOY_KERNELS=1 $PIP install -e . --no-build-isolation
fi

echo
echo ">>> Done. Verifying core imports..."
$PY -c "import torch, fast_hadamard_transform, flatquant; print('OK: flatquant ready (torch', torch.__version__, ')')"
echo
echo "Next: download a model + datasets, e.g.:"
echo "    bash scripts/download_assets.sh Qwen/Qwen2.5-7B-Instruct"
echo "Then run, e.g.:"
echo "    bash scripts/qwen-2.5-instruct/qwen-2.5-instruct-7b/w4a4kv4.sh"
