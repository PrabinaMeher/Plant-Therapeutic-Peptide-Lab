# Plant Therapeutic Peptide Lab (PTPLAB)
End-to-end pipeline for plant-based bioactive peptide discovery: PLM-based classification (7 bioactivity classes) → ESMFold structure prediction → HADDOCK3 docking → GROMACS MD → MM-PBSA binding free energy. Built by ICAR-IASRI.

## Setup on a new machine

```bash
git clone git@github.com:PrabinaMeher/Plant-Therapeutic-Peptide-Lab.git
cd Plant-Therapeutic-Peptide-Lab
bash setup.sh
conda activate peptide_sklearn
streamlit run app.py
```

Notes:
- Requires conda and an NVIDIA GPU with driver + CUDA toolkit compatible packages.
- The `esmfold_gpu` environment needs the CUDA toolkit and CCCL headers installed separately if not already present:
```bash
  conda activate esmfold_gpu
  conda install -c nvidia cuda-toolkit=12.1 cuda-cccl=12.1 -y
  conda env config vars set CUDA_HOME=$CONDA_PREFIX -n esmfold_gpu
```
- The repo's `openfold/` folder is a fixed version compatible with the ESMFold checkpoint (uses plain `Linear` layers, not `PointProjection`) — do not replace it with a fresh clone of upstream openfold, which uses an incompatible newer API.
