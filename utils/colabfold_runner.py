import os
import subprocess

def fold_peptide(sequence, output="peptide.pdb"):
    """
    Generate peptide structure using ColabFold
    """

    fasta_file = "query.fasta"

    # Write FASTA
    with open(fasta_file, "w") as f:
        f.write(">peptide\n")
        f.write(sequence)

    out_dir = "colabfold_out"
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        "colabfold_batch",
        fasta_file,
        out_dir,
        "--num-models", "1",
        "--num-recycle", "1"
    ]

    subprocess.run(cmd, check=True)

    # Default ColabFold output
    pdb_path = os.path.join(
        out_dir,
        "query_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb"
    )

    # Rename to desired output
    os.rename(pdb_path, output)

    return output
