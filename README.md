# Plant Therapeutic Peptide Lab (PTPLab)

An end to end bioactive peptide discovery platform: sequence based bioactivity prediction, ESMFold structure prediction, HADDOCK3 docking, GROMACS molecular dynamics, and gmx_MMPBSA binding free energy analysis, all wrapped in a Streamlit web app.

## Pipeline Overview

1. Peptide Prediction: Protein Language Models (ProtT5, ProtAlbert) screen candidate peptides across 7 bioactivity classes (ABP, ACP, AFP, AHP, AIP, APP, AVP).
2. Structure Prediction: ESMFold predicts the 3D structure of the top candidate peptide.
3. Receptor Preparation: download a receptor from RCSB PDB or upload your own; ligands, water, and DNA are cleaned automatically.
4. Molecular Docking: HADDOCK3 docks the peptide against the receptor and produces a HADDOCK score.
5. Molecular Dynamics: GROMACS simulates the docked complex (solvation, ions, energy minimization, NVT, NPT, production MD) and reports RMSD, RMSF, and radius of gyration.
6. Binding Free Energy: gmx_MMPBSA computes the final binding free energy (deltaG).

## Requirements

- Linux (tested on Ubuntu)
- Conda (Miniconda or Anaconda)
- NVIDIA GPU with a recent driver, for ESMFold structure prediction
- Roughly 30 to 60 minutes and several GB of disk space for the first environment setup

## Cloning the Repository

Anyone who only wants to run the app can clone with the plain HTTPS URL. No GitHub account or SSH key is required for this, since the repository is public.

```bash
git clone https://github.com/PrabinaMeher/Plant-Therapeutic-Peptide-Lab.git
cd Plant-Therapeutic-Peptide-Lab
```

SSH access (`git@github.com:...`) is only needed if you plan to push changes back to the repository as a collaborator. That requires your own SSH key added to your own GitHub account, and write access to the repository.

## One Command Setup

```bash
bash setup.sh
```

This script will:

- Create all four conda environments from the files in `environments/`
- Install gmx_MMPBSA and a compiler toolchain into `mmpbsa_env`
- Register the repository's local `openfold` package inside `esmfold_gpu`
- Install the CUDA toolkit and CCCL headers needed by deepspeed and openfold
- Verify that openfold imports correctly and that PyTorch can see the GPU
- Install Ollama and pull the `llama3.2:3b` model for the in app assistant

Run it once from the repository root and let it finish without interrupting it, since some steps depend on earlier ones completing first.

## Conda Environments

| Environment | Purpose |
|---|---|
| peptide_sklearn | Runs the Streamlit app itself (streamlit, scikit learn, plotly, py3Dmol, pandas, numpy) |
| esmfold_gpu | ESMFold structure prediction (PyTorch, fair esm, local openfold, deepspeed, CUDA toolkit) |
| haddock_env | HADDOCK3 molecular docking |
| mmpbsa_env | GROMACS and gmx_MMPBSA for molecular dynamics and binding free energy |

## Running the App

```bash
conda activate peptide_sklearn
streamlit run app.py
```

Open the Local URL that Streamlit prints, usually `http://localhost:8501`.

## Ollama Setup (Ask Assistant feature)

The in app "Ask Assistant" chat uses a local Ollama model. `setup.sh` installs this automatically, but if you need to do it manually:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2:3b
```

Ollama must be running as a background service before using the assistant:

```bash
ollama serve
```

Run this in its own terminal, separate from the Streamlit app. If Ollama is not running, the assistant automatically falls back to a built in FAQ search instead of failing.

## Dependencies by Component

### Peptide Prediction (peptide_sklearn)
- streamlit
- pandas, numpy
- scikit learn
- plotly
- py3Dmol
- ProtT5, ProtAlbert (via transformers/torch, loaded at runtime)

### Structure Prediction (esmfold_gpu)
- torch (CUDA build matching the installed driver)
- fair esm
- local `openfold/` package (already included in this repository; do not replace it with a fresh clone of upstream openfold, since upstream uses a newer, incompatible API for its attention module)
- deepspeed
- CUDA toolkit and CCCL headers (installed by `setup.sh`)

### Molecular Docking (haddock_env)
- haddock3
- streamlit, scikit learn, plotly (for shared imports where relevant)

### Molecular Dynamics (mmpbsa_env)
- GROMACS (gmx)
- gmx_MMPBSA
- gcc_linux-64, gxx_linux-64 (compiler toolchain gmx_MMPBSA needs at install time)

## Known Issues and Fixes Already Applied

- The local `openfold/` folder in this repository is a specific version compatible with the ESMFold checkpoint's expected parameter names. Do not replace it with a fresh clone of the upstream openfold repository, since newer versions restructure the attention module and will fail to load the pretrained weights.
- `esmfold_runner.py` stubs the `attn_core_inplace_cuda` module before importing openfold, so the compiled CUDA kernel does not need to be built.
- The MD trajectory analysis pipeline runs `gmx trjconv -pbc nojump` before centering and fitting, to avoid periodic image jump artifacts that otherwise show up as sawtooth patterns in RMSD and radius of gyration plots.
- If `gmx mdrun` fails with a fatal error about allocating an extremely large number of elements, this is usually an integer overflow triggered by a mismatch between requested CPU threads and the box or PME grid size, not literally insufficient memory. Try reducing `-ntomp` to match the actual core count reported by `nproc`.

## Troubleshooting

If ESMFold fails with `CUDA_HOME does not exist`, the CUDA toolkit was not installed into `esmfold_gpu`, or `setup.sh` exited early during environment creation. Run:

```bash
conda activate esmfold_gpu
conda install -c nvidia cuda-toolkit=12.1 cuda-cccl=12.1 -y
conda env config vars set CUDA_HOME=$CONDA_PREFIX -n esmfold_gpu
conda deactivate
conda activate esmfold_gpu
```

If `gmx_MMPBSA` is reported as not found in `mmpbsa_env`, install it along with the compiler toolchain it needs:

```bash
conda activate mmpbsa_env
conda install -c conda-forge gcc_linux-64 gxx_linux-64 -y
pip install gmx_MMPBSA
```

## References

| Tool | Link |
|---|---|
| HADDOCK3 | https://github.com/haddocking/haddock3 |
| ESMFold | https://esmatlas.com |
| GROMACS | https://www.gromacs.org |
| gmx_MMPBSA | https://github.com/Valdes-Tresanco-MS/gmx_MMPBSA |

## Team

- Dr. Prabina Kumar Meher, Senior Scientist, ICAR-IASRI, New Delhi, India
- Dr. Upendra Kumar Pradhan, Senior Scientist, ICAR-IASRI, New Delhi, India
- Shubham Kumar, Young Professional II, ICAR-IASRI, New Delhi, India
- Aanchal Gupta, Project Associate I, ICAR-IASRI, New Delhi, India
