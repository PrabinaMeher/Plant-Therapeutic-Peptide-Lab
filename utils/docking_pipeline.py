import os
import requests
from utils.druglikeness import evaluate_peptide
from utils.toxicity import check_toxicity
from utils.structure import prepare_peptide
from esmfold_runner import fold_sequence  # <- your CPU/GPU-safe ESMFold

# STEP 1 — Drug Check
def check_peptide(sequence):
    drug = evaluate_peptide(sequence)
    tox = check_toxicity(sequence)
    return drug, tox


# STEP 2 — Run ESMFold
def run_esmfold(sequence, out_dir="esmfold_output", use_gpu=False, chunk_size=64):
    """
    Generate peptide PDB using ESMFold.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Prepare a temporary PDB file path
    pdb_file = os.path.join(out_dir, "peptide.pdb")

    # Generate structure
    fold_sequence(sequence, pdb_file, use_gpu=use_gpu, chunk_size=chunk_size)

    return pdb_file


# STEP 3 — Download Receptor from PDB
def download_receptor(pdb_id, out_dir="receptor"):
    os.makedirs(out_dir, exist_ok=True)
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    r = requests.get(url)

    pdb_file = os.path.join(out_dir, f"{pdb_id}.pdb")
    with open(pdb_file, "wb") as f:
        f.write(r.content)

    return pdb_file


# STEP 4 — Run DiffDock
def run_diffdock(peptide_pdb, receptor_pdb, out_dir="docking_output"):
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        "python",
        "run_diffdock.py",
        "--protein",
        receptor_pdb,
        "--ligand",
        peptide_pdb,
        "--out_dir",
        out_dir
    ]

    # Run DiffDock as a subprocess
    import subprocess
    subprocess.run(cmd, check=True)

    return out_dir


# FULL DOCKING PIPELINE
def run_full_docking(sequence, pdb_id, use_gpu=False):
    """
    Wrapper function for Streamlit app.
    Returns dict with status, docking output, and optional pLDDT info.
    """
    try:
        # Step 1: Drug & toxicity check
        drug, tox = check_peptide(sequence)

        # Step 2: Fold peptide
        peptide_pdb = run_esmfold(sequence, use_gpu=use_gpu)

        # Step 3: Download receptor
        receptor_pdb = download_receptor(pdb_id)

        # Step 4: Dock
        docking_output_dir = run_diffdock(peptide_pdb, receptor_pdb)

        result = {
            "status": "success",
            "docking_output": {"best_pose": os.path.join(docking_output_dir, "best_pose.pdb")},
            "structure": peptide_pdb,
            "drug_score": drug,
            "toxicity_score": tox,
            # pLDDT not computed here; could integrate if needed
        }

    except Exception as e:
        result = {"status": "error", "message": str(e)}

    return result
