from deeprank_gnn.Graph import Graph
import os


def build_graph(complex_pdb, output_hdf5):

    if not os.path.exists(complex_pdb):
        raise RuntimeError("Complex PDB file not found")

    print("Generating DeepRank-GNN graph...")

    # Create graph object
    graph = Graph()

    # Build graph from complex structure
    graph.create_graph(
        pdb_path=complex_pdb,
        out_file=output_hdf5
    )

    if not os.path.exists(output_hdf5):
        raise RuntimeError("Graph generation failed")

    print("Graph successfully generated:", output_hdf5)

    return output_hdf5
