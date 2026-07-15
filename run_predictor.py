from predictor import PlantTherapeuticPredictor

_predictor = None


def get_predictor():
    global _predictor

    if _predictor is None:
        print("Loading predictor (ProtT5 + ML models)...")
        _predictor = PlantTherapeuticPredictor()
        print("Predictor Ready")

    return _predictor


def run_prediction(fasta_path, task):

    global _predictor
    predictor = get_predictor()

    result_df = predictor.predict(fasta_path, task)

    predictor.unload()
    _predictor = None

    return result_df
