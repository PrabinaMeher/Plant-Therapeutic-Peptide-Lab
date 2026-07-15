import os

def prepare_peptide(sequence, output="peptide.fasta"):

    with open(output, "w") as f:
        f.write(">peptide\n")
        f.write(sequence)

    return output
