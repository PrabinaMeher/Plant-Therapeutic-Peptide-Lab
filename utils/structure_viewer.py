import py3Dmol
from stmol import showmol


def view_structure(pdb_path, height=500, width=800):

    with open(pdb_path) as f:
        pdb_data = f.read()

    viewer = py3Dmol.view(width=width, height=height)

    viewer.addModel(pdb_data, "pdb")

    # Protein style
    viewer.setStyle(
        {"chain": "A"},
        {"cartoon": {"color": "spectrum"}}
    )

    # Peptide / ligand style
    viewer.setStyle(
        {"chain": "B"},
        {"stick": {"colorscheme": "greenCarbon"}}
    )

    viewer.zoomTo()
    viewer.setBackgroundColor("white")

    showmol(viewer, height=height, width=width)
