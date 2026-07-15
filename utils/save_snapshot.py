import py3Dmol


def save_docking_snapshot(receptor_pdb, ligand_pdb, output_png="docking.png"):

    view = py3Dmol.view(width=900, height=550)

    with open(receptor_pdb) as f:
        view.addModel(f.read(), "pdb")

    with open(ligand_pdb) as f:
        view.addModel(f.read(), "pdb")

    view.setStyle({"model": 0}, {"cartoon": {"color": "lightgray"}})
    view.setStyle({"model": 1}, {"stick": {"colorscheme": "cyanCarbon"}})

    view.zoomTo()

    png = view.png()

    with open(output_png, "wb") as f:
        f.write(png)

    return output_png
