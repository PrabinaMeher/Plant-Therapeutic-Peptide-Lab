import os
from deeprank_gnn.NeuralNet import NeuralNet

MODEL_PATH = "models/deeprank_model.pth"


def predict_affinity(graph_file):
    """
    Predict binding affinity using DeepRank-GNN model
    """

    if not os.path.exists(graph_file):
        raise RuntimeError("Graph file not found")

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError("DeepRank model file not found")

    try:
        print("Running DeepRank-GNN affinity prediction...")

        model = NeuralNet(
            database=graph_file,
            model=MODEL_PATH,
            task="regress"
        )

        score = model.predict()

        # Sometimes DeepRank returns list
        if isinstance(score, (list, tuple)):
            score = score[0]

        print("Predicted affinity:", score)

        return float(score)

    except Exception as e:
        raise RuntimeError(f"GNN scoring failed: {str(e)}")
