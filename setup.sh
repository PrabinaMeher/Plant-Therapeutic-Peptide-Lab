#!/bin/bash
set -e
echo "=== PTPLab Environment Setup ==="

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

source "$(conda info --base)/etc/profile.d/conda.sh"

echo "--- Creating conda environments ---"
conda env create -f environments/peptide_sklearn.yml || echo "peptide_sklearn already exists, skipping"
conda env create -f environments/esmfold_gpu.yml || echo "esmfold_gpu already exists, skipping"
conda env create -f environments/haddock_env.yml || echo "haddock_env already exists, skipping"
conda env create -f environments/mmpbsa_env.yml || echo "mmpbsa_env already exists, skipping"
echo "--- Installing gmx_MMPBSA and compiler toolchain in mmpbsa_env ---"
conda activate mmpbsa_env
conda install -c conda-forge gcc_linux-64 gxx_linux-64 -y
pip install gmx_MMPBSA
conda deactivate

echo "--- Registering local openfold package in esmfold_gpu ---"
conda activate esmfold_gpu
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
echo "$REPO_ROOT" > "$SITE_PACKAGES/ptplab_repo.pth"

echo "--- Installing CUDA toolkit for deepspeed/openfold compatibility ---"
conda install -c nvidia cuda-toolkit=12.1 cuda-cccl=12.1 -y
conda env config vars set CUDA_HOME=$CONDA_PREFIX -n esmfold_gpu
conda deactivate
conda activate esmfold_gpu

echo "--- Verifying esmfold_gpu setup ---"
python -c "
import sys, types
stub = types.ModuleType('attn_core_inplace_cuda')
stub.forward_ = lambda *a, **k: None
stub.backward_ = lambda *a, **k: None
sys.modules['attn_core_inplace_cuda'] = stub

import openfold
import torch
print('openfold OK:', openfold.__file__)
print('torch:', torch.__version__, 'CUDA available:', torch.cuda.is_available())
"

conda deactivate
echo "--- Installing Ollama for Ask Assistant feature ---"
if ! command -v ollama &> /dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "Ollama already installed, skipping"
fi
ollama pull llama3.2:3b
echo "=== Setup complete ==="
echo "Run the app with: conda activate peptide_sklearn && streamlit run app.py"
echo "Note: run '''ollama serve''' in a separate terminal before using the Ask Assistant feature."
