from Bio.PDB import PDBParser
import numpy as np

def detect_hydrogen_bonds(receptor_pdb, ligand_pdb, cutoff=3.5):

    parser = PDBParser(QUIET=True)

    rec = parser.get_structure("rec", receptor_pdb)
    lig = parser.get_structure("lig", ligand_pdb)

    bonds = []

    for atom1 in rec.get_atoms():
        for atom2 in lig.get_atoms():

            dist = atom1 - atom2

            if dist < cutoff:
                bonds.append({
                    "rec_atom": atom1.get_name(),
                    "lig_atom": atom2.get_name(),
                    "distance": round(dist,2)
                })

    return bonds
