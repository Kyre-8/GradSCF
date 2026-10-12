#!/bin/bash
# Submit from a GradSCF checkout under ~/private/homefile on jy_gpu_2.
#SBATCH --partition=home
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --job-name=neural-xc-memory
#SBATCH --output=neural-xc-memory-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

neural_python="$HOME/private/homefile/conda-envs/jax_scf/bin/python"
neural_result="${NEURAL_XC_MEMORY_OUTPUT:-$PWD/results/$SLURM_JOB_ID}"
neural_cache="/tmp/neural-xc-memory-$SLURM_JOB_ID"
mkdir -p "$neural_result" "$neural_cache/cuda" "$neural_cache/jax"
export PATH="$(dirname "$neural_python"):$PATH"

export PYTHONPATH="$PWD/src"
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_CACHE_PATH="$neural_cache/cuda"
export JAX_COMPILATION_CACHE_DIR="${NEURAL_XC_JAX_CACHE_DIR:-$neural_cache/jax}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NEURAL_XC_MEMORY_OUTPUT="$neural_result"

hostname
nvidia-smi > "$neural_result/nvidia-smi.txt"
printf '%s\n' "$JAX_COMPILATION_CACHE_DIR" > "$neural_result/jax-cache-path.txt"
"$neural_python" -m pip freeze > "$neural_result/environment.txt"
if [[ "${NEURAL_XC_BUILD_NATIVE:-0}" == 1 ]]; then
    if [[ -f /opt/rh/gcc-toolset-12/enable ]]; then
        source /opt/rh/gcc-toolset-12/enable
    fi
    JAX_PLATFORMS=cpu "$neural_python" -m gradscf.integrals._native.build \
        --build-dir "$PWD/build-native-linux" --jobs 2 \
        > "$neural_result/native-build.log" 2>&1
fi
find src -name '*.py' -type f -print0 | sort -z | xargs -0 sha256sum \
    > "$neural_result/source-sha256.txt"
sha256sum examples/training/neural_xc/h2_implicit_memory.py \
    examples/training/neural_xc/gpu_memory.py \
    src/gradscf/integrals/_native/lib/libgradscf_integrals.so \
    > "$neural_result/benchmark-sha256.txt"
"$neural_python" examples/training/neural_xc/h2_implicit_memory.py
