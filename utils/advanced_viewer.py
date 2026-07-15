import os
import py3Dmol
from stmol import showmol


def view_docking_complex(receptor_pdb, ligand_pdb, width=900, height=550):

    if not os.path.exists(receptor_pdb):
        raise RuntimeError("Receptor PDB not found")

    if not os.path.exists(ligand_pdb):
        raise RuntimeError("Ligand PDB not found")

    view = py3Dmol.view(width=width, height=height)

    # Load receptor
    with open(receptor_pdb) as f:
        receptor_data = f.read()
        view.addModel(receptor_data, "pdb")

    # Load ligand
    with open(ligand_pdb) as f:
        ligand_data = f.read()
        view.addModel(ligand_data, "pdb")

    # Protein cartoon style
    view.setStyle(
        {"model": 0},
        {"cartoon": {"color": "lightgray"}}
    )

    # Ligand stick style
    view.setStyle(
        {"model": 1},
        {"stick": {"colorscheme": "cyanCarbon", "radius": 0.25}}
    )

    # Highlight residues near ligand
    view.setStyle(
        {"model": 0, "within": {"distance": 6, "model": 1}},
        {"stick": {"color": "orange"}}
    )

    # Add surface
    view.addSurface(
        py3Dmol.VDW,
        {"opacity": 0.6, "color": "white"},
        {"model": 0}
    )

    view.zoomTo()
    view.setBackgroundColor("white")

    showmol(view, height=height, width=width)
