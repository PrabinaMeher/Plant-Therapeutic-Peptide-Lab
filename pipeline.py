#!/usr/bin/env python3

import os
import subprocess
import pandas as pd
import requests
import re

from utils.druglikeness import evaluate_peptide
from utils.toxicity import check_toxicity
from run_predictor import run_prediction
from Bio.SeqUtils.ProtParam import ProteinAnalysis

from Bio import SeqIO

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

def clean_sequence(seq):
    seq = str(seq).upper().strip()
    seq = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', seq)
    return seq

# CONFIG


THERAPEUTIC_CLASSES = ["Antibacterial (ABP)", "Anticancer (ACP)", "Antifungal (AFP)", "Anti-HIV (AHP)", "Anti-inflammatory (AIP)", "Antiparasitic (APP)", "Antiviral (AVP)"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = BASE_DIR

ESMFOLD_PYTHON = os.environ.get(
    "ESMFOLD_PYTHON",
    "/home/user/anaconda3/envs/esmfold_gpu/bin/python"
)

DOCK_PYTHON = os.environ.get(
    "DOCK_PYTHON",
    "/home/user/anaconda3/envs/esmfold_diffdock/bin/python"
)



#  PREDICTION


def predict_peptides(fasta_path, task):

    print("\n Prediction ")

    clean_records = []

    for record in SeqIO.parse(fasta_path, "fasta"):
        seq = clean_sequence(record.seq)

        if len(seq) == 0:
            continue

        clean_records.append((record.id, seq))

    if not clean_records:
        raise RuntimeError("No valid sequences after cleaning")

    # Save cleaned FASTA
    cleaned_fasta = fasta_path + "_clean.fasta"

    with open(cleaned_fasta, "w") as f:
        for header, seq in clean_records:
            f.write(f">{header}\n{seq}\n")

    # Run prediction
    df = run_prediction(cleaned_fasta, task)

    if df is None or df.empty:
        raise RuntimeError("Prediction returned empty result")

    print("Prediction completed")

    return df

def evaluate_candidate(sequence):

    print("\n Drug/Toxicity Evaluation ")

    # CLEAN SEQUENCE FIRST
    sequence = clean_sequence(sequence)

    if len(sequence) == 0:
        raise ValueError("Invalid sequence after cleaning")

    # Optional: enforce peptide length
    if len(sequence) > 150:
        print("⚠️ Warning: Long sequence may not be suitable for peptide analysis")

    #  SAFE ANALYSIS
    analysis = ProteinAnalysis(sequence)

    length = len(sequence)
    mw = analysis.molecular_weight()
    charge = analysis.charge_at_pH(7.0)
    pI = analysis.isoelectric_point()
    gravy = analysis.gravy()
    aromatic = analysis.aromaticity()

    # Drug score
    drug_raw = evaluate_peptide(sequence)
    drug_score = float(drug_raw.get("score", 0)) if isinstance(drug_raw, dict) else float(drug_raw)

    # Toxicity
    tox_raw = check_toxicity(sequence)
    tox_score = float(tox_raw.get("score", 0)) if isinstance(tox_raw, dict) else float(tox_raw)

    rules = {
        "Length_OK": 5 <= length <= 50,
        "Charge_OK": -2 <= charge <= 6,
        "Hydrophobicity_OK": -1 <= gravy <= 1.5,
        "Aromatic_OK": aromatic <= 0.5
    }

    return {
        "length": length,
        "mw": round(mw, 2),
        "charge": round(charge, 2),
        "pI": round(pI, 2),
        "hydrophobicity": round(gravy, 3),
        "aromatic": round(aromatic, 3),
        "rules": rules,
        "drug_score": round(drug_score, 3),
        "toxicity_score": round(tox_score, 3)
    }


#  STRUCTURE


#def fold_peptide(sequence, output_file="peptide_structure.pdb"):

#    print("\n  Peptide Folding ")

#    output_path = os.path.join(PROJECT_ROOT, output_file)

#    esm_script = os.path.join(BASE_DIR, "docking", "esmfold_runner.py")

#    cmd = [
#        ESMFOLD_PYTHON,
#        esm_script,
#        sequence,
#        output_path,
#        "1"
#    ]

#    subprocess.run(cmd, check=True)

 #   if not os.path.exists(output_path):
 #       raise RuntimeError("ESMFold failed")

 #   return output_path

def fold_peptide(sequence, output_file=None):
    import tempfile

    print("\n Peptide Folding ")

    #  Fix 1: Always use unique temp directory, never fixed filename 
    if output_file is None:
        out_dir = tempfile.mkdtemp(prefix="ptplab_esm_")
        output_path = os.path.join(out_dir, "esmfold_peptide.pdb")
    else:
        output_path = os.path.abspath(output_file)

    esm_script = os.path.join(BASE_DIR, "docking", "esmfold_runner.py")

    cmd = [
        ESMFOLD_PYTHON,
        esm_script,
        sequence,
        output_path,
        "1"
    ]

    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(result.stderr)
        
    if result.stderr:
        print("STDERR:", result.stderr[:500])

    if not os.path.exists(output_path):
        raise RuntimeError(f"ESMFold failed — output file not created: {output_path}")

    #  Fix 2: Validate the PDB before returning 
    lines = open(output_path).readlines()
    atom_lines = [l for l in lines if l.startswith("ATOM")]

    if len(atom_lines) == 0:
        raise RuntimeError(
            f"ESMFold output has no ATOM records.\n"
            f"File contents: {open(output_path).read()[:300]}"
        )

    #  Fix 3: Check for truncated lines 
    
    short = [l for l in atom_lines if len(l.rstrip()) < 60]
    if short:
        raise RuntimeError(
            f"ESMFold PDB has {len(short)} malformed lines (< 60 chars).\n"
            f"Example: {short[0]!r}"
        )

    #  Fix 4: Ensure chain A and proper TER/END 
    fixed = []
    serial = 1
    for line in lines:
        if line.startswith("ATOM") and len(line.rstrip()) >= 60:
            line = line[:21] + "A" + line[22:]          # ensure chain A
            line = f"ATOM  {serial:5d}{line[11:]}"       # renumber serials
            serial += 1
            fixed.append(line if line.endswith("\n") else line + "\n")
        elif line.startswith("HETATM") and len(line.rstrip()) >= 54:
            fixed.append(line if line.endswith("\n") else line + "\n")
        elif line.startswith(("REMARK","MODEL","ENDMDL","HEADER","TITLE","CRYST1")):
            fixed.append(line if line.endswith("\n") else line + "\n")
        elif line.startswith(("TER","END")):
            continue  # add our own below

    fixed.append("TER\n")
    fixed.append("END\n")
    open(output_path, "w").writelines(fixed)

    print(f"ESMFold complete: {len(atom_lines)} atoms → {output_path}")
    return os.path.abspath(output_path)

#  RECEPTOR


def download_receptor(pdb_id):

    print("\n  Download Receptor ")

    receptor_dir = os.path.join(PROJECT_ROOT, "receptors")
    os.makedirs(receptor_dir, exist_ok=True)

    pdb_id = pdb_id.upper()
    receptor_path = os.path.join(receptor_dir, f"{pdb_id}.pdb")

    if os.path.exists(receptor_path):
        return receptor_path

    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"

    r = requests.get(url, timeout=600)

    if r.status_code != 200:
        raise RuntimeError(f"Failed to download receptor: {pdb_id}")

    with open(receptor_path, "wb") as f:
        f.write(r.content)

    return receptor_path



# CLEAN PDB (IMPORTANT)


def clean_pdb(input_path, output_path):
    with open(input_path) as f, open(output_path, "w") as out:
        for line in f:
            if line.startswith("ATOM"):
                out.write(line)



#  DOCKING (FINAL)


def run_docking(peptide_pdb, receptor_pdb):

    print("\n  Docking ")

    docking_dir = os.path.abspath(os.path.join(PROJECT_ROOT, "docking_output"))
    os.makedirs(docking_dir, exist_ok=True)

    peptide_pdb = os.path.abspath(peptide_pdb)
    receptor_pdb = os.path.abspath(receptor_pdb)

    #  Clean receptor (VERY IMPORTANT)
    clean_receptor = receptor_pdb.replace(".pdb", "_clean.pdb")
    clean_pdb(receptor_pdb, clean_receptor)
    receptor_pdb = clean_receptor

    diffdock_script = os.path.join(BASE_DIR, "docking", "diffdock_runner.py")

    cmd = [
        DOCK_PYTHON,
        diffdock_script,
        "--protein_path", receptor_pdb,   
        "--ligand", peptide_pdb,
        "--out_dir", docking_dir
    ]

    print("Running DiffDock...")
    print("Command:", " ".join(cmd))

    result = subprocess.run(
        cmd,
        cwd=BASE_DIR,
        capture_output=True,
        text=True
    )

    print("\n STDOUT \n", result.stdout)
    print("\n STDERR \n", result.stderr)

    if result.returncode != 0:
        raise RuntimeError("DiffDock crashed")

    #  Detect output files
    files = os.listdir(docking_dir)
    print("Files in docking dir:", files)

    pdb_files = [
        os.path.join(docking_dir, f)
        for f in files if f.endswith(".pdb")
    ]

    if not pdb_files:
        raise RuntimeError("Docking ran but no PDB files found")

    #  Select best pose
    def extract_score(f):
        match = re.search(r"confidence([0-9.]+)", f)
        return float(match.group(1)) if match else 0

    best_pose = max(pdb_files, key=extract_score)

    print("Selected best pose:", best_pose)

    return best_pose
