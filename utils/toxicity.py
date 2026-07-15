import subprocess
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)


def generate_structure(sequence, output_file="toxicity_peptide.pdb"):

    script = os.path.join(PROJECT_ROOT, "docking", "esmfold_runner.py")
    output_path = os.path.join(PROJECT_ROOT, output_file)

    cmd = [
        "conda",
        "run",
        "-n",
        "esmfold_gpu",
        "python",
        script,
        sequence,
        output_path
    ]

    subprocess.run(cmd, check=True)

    return output_path


def check_toxicity(sequence, use_structure=False):
    """
    Example placeholder toxicity function
    """

    pdb_file = None

    # Only generate structure if required
    if use_structure:
        pdb_file = generate_structure(sequence)

    # Your toxicity logic here
    toxicity_score = 0.1

    return toxicity_score
