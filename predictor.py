#!/usr/bin/env python3

import os
import gc
import sys
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from Bio import SeqIO

from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM, Bidirectional, SimpleRNN, InputLayer
from tensorflow.keras.mixed_precision import Policy

from embeddings import ProtT5Embedder
from transformers import AlbertTokenizer, AlbertModel
import torch


# Fix numpy serialization issues
sys.modules["numpy._core"] = np.core



# GPU CONFIGURATION


# Disable GPU for TensorFlow (prevents CUDA conflict)
tf.config.set_visible_devices([], "GPU")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch device: {DEVICE}")

torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = True



# GLOBAL MODEL CACHE


_GLOBAL_MODELS = {
    "protT5": None,
    "protAlbert": None
}



# PATCHED KERAS LAYERS


_custom_objs = tf.keras.utils.get_custom_objects()

if "Custom>InputLayerPatched" not in _custom_objs:
    @tf.keras.utils.register_keras_serializable()
    class InputLayerPatched(InputLayer):

        def __init__(self, *args, **kwargs):

            if "shape" in kwargs and "batch_input_shape" not in kwargs:
                shape = kwargs.pop("shape")
                kwargs["batch_input_shape"] = (None,) + tuple(shape)

            if "batch_shape" in kwargs and "batch_input_shape" not in kwargs:
                kwargs["batch_input_shape"] = kwargs.pop("batch_shape")

            super().__init__(*args, **kwargs)
else:
    InputLayerPatched = _custom_objs["Custom>InputLayerPatched"]


if "Custom>SimpleRNNPatched" not in _custom_objs:
    @tf.keras.utils.register_keras_serializable()
    class SimpleRNNPatched(SimpleRNN):

        def __init__(self, *args, time_major=None, **kwargs):
            super().__init__(*args, **kwargs)
else:
    SimpleRNNPatched = _custom_objs["Custom>SimpleRNNPatched"]


if "Custom>LSTMPatched" not in _custom_objs:
    @tf.keras.utils.register_keras_serializable()
    class LSTMPatched(LSTM):

        def __init__(self, *args, **kwargs):
            kwargs.pop("time_major", None)
            super().__init__(*args, **kwargs)
else:
    LSTMPatched = _custom_objs["Custom>LSTMPatched"]


if "Custom>BiLSTMPatched" not in _custom_objs:
    @tf.keras.utils.register_keras_serializable()
    class BiLSTMPatched(Bidirectional):

        def __init__(self, layer, *args, **kwargs):

            if isinstance(layer, LSTM):
                layer = LSTMPatched(**layer.get_config())

            super().__init__(layer, *args, **kwargs)
else:
    BiLSTMPatched = _custom_objs["Custom>BiLSTMPatched"]



# PROTALBERT EMBEDDER


class ProtAlbertEmbedder:

    def __init__(self):

        global _GLOBAL_MODELS

        if _GLOBAL_MODELS["protAlbert"] is None:

            print("Loading ProtAlbert model...")

            tokenizer = AlbertTokenizer.from_pretrained(
                "Rostlab/prot_albert",
                do_lower_case=False
            )

            model = AlbertModel.from_pretrained("Rostlab/prot_albert")

            model.to(DEVICE)
            model.eval()

            _GLOBAL_MODELS["protAlbert"] = (tokenizer, model)

        self.tokenizer, self.model = _GLOBAL_MODELS["protAlbert"]

    def embed(self, sequences):

        processed = [" ".join(list(seq)) for seq in sequences]

        inputs = self.tokenizer(
            processed,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(DEVICE)

        with torch.no_grad():
            outputs = self.model(**inputs)

        embeddings = outputs.last_hidden_state.mean(dim=1)

        return embeddings.detach().cpu().numpy()



# MAIN PREDICTOR


class PlantTherapeuticPredictor:

    MAX_BATCH_SIZE = 1000
    
    def __init__(self):

        print("Initializing Predictor...")

        self.protT5_embedder = ProtT5Embedder()
        self.protAlbert_embedder = ProtAlbertEmbedder()

        # embedding cache
        self._cached_sequences_t5 = None
        self._cached_sequences_albert = None

        self._cached_t5 = None
        self._cached_albert = None

        self.load_models()

        print("Predictor Ready")



# LOAD MODELS


    def load_models(self):

        print("Loading Deep Learning models...")

        custom_layers = {
            "InputLayer": InputLayerPatched,
            "SimpleRNN": SimpleRNNPatched,
            "LSTM": LSTMPatched,
            "Bidirectional": BiLSTMPatched,
            "DTypePolicy": Policy
        }

        self.abp_model = load_model(
            "models/abp_rnn.h5",
            custom_objects=custom_layers,
            compile=False
        )

        self.afp_model = load_model(
            "models/afp_rnn.h5",
            custom_objects=custom_layers,
            compile=False
        )

        self.ahp_model = load_model(
            "models/ahp_ann.h5",
            custom_objects=custom_layers,
            compile=False
        )

        self.app_model = load_model(
            "models/app_bilstm.h5",
            custom_objects=custom_layers,
            compile=False
        )

        self.avp_model = load_model(
            "models/avp_bilstm.h5",
            custom_objects=custom_layers,
            compile=False
        )

        print("Loading ML models...")

        acp_data = joblib.load("models/acp_gbdt.pkl")
        self.acp_model = acp_data["model"]
        self.acp_scaler = acp_data["scaler"]

        self.aip_model = joblib.load("models/aip_svm_fixed.pkl")

        self.scalers = {}

        for name in ["abp", "afp", "ahp", "app", "avp"]:

            path = f"scalers/{name}_scaler.pkl"

            self.scalers[name] = (
                joblib.load(path) if os.path.exists(path) else None
            )

        print("All models loaded successfully")



# FASTA READER


    def _read_fasta(self, fasta_path):

        records = list(SeqIO.parse(fasta_path, "fasta"))

        if len(records) == 0:
            raise ValueError("No sequences found.")

        if len(records) > self.MAX_BATCH_SIZE:
            raise ValueError(
                f"Too many sequences ({len(records)}). Max allowed: {self.MAX_BATCH_SIZE}"
            )

        protein_ids = [r.id for r in records]
        sequences = [str(r.seq).upper() for r in records]

        return protein_ids, sequences



# EMBEDDINGS


    def _get_protT5(self, sequences):

        if self._cached_sequences_t5 != sequences:

            print("Generating ProtT5 embeddings (GPU)...")

            self._cached_t5 = self.protT5_embedder.embed(sequences)
            self._cached_sequences_t5 = sequences

        return self._cached_t5


    def _get_protAlbert(self, sequences):

        if self._cached_sequences_albert != sequences:

            print("Generating ProtAlbert embeddings (GPU)...")

            self._cached_albert = self.protAlbert_embedder.embed(sequences)
            self._cached_sequences_albert = sequences

        return self._cached_albert



# PREDICTION


    def predict(self, fasta_path, task):

        task = task.upper()

        protein_ids, sequences = self._read_fasta(fasta_path)

        results = pd.DataFrame({
            "protein_id": protein_ids,
            "sequence": sequences
        })

        if task == "AIP":

            X = self._get_protAlbert(sequences)
            scores = self.aip_model.predict_proba(X)[:, 1]

        else:

            X = self._get_protT5(sequences)

            if task == "ACP":

                X_scaled = self.acp_scaler.transform(X)
                scores = self.acp_model.predict_proba(X_scaled)[:, 1]

            elif task == "ABP":

                X_input = self._prepare_rnn_input(X, "abp")
                scores = self.abp_model.predict(X_input, verbose=0).flatten()

            elif task == "AFP":

                X_input = self._prepare_rnn_input(X, "afp")
                scores = self.afp_model.predict(X_input, verbose=0).flatten()

            elif task == "AHP":

                X_input = self._prepare_dense_input(X, "ahp")
                scores = self.ahp_model.predict(X_input, verbose=0).flatten()

            elif task == "APP":

                X_input = self._prepare_rnn_input(X, "app")
                scores = self.app_model.predict(X_input, verbose=0).flatten()

            elif task == "AVP":

                X_input = self._prepare_rnn_input(X, "avp")
                scores = self.avp_model.predict(X_input, verbose=0).flatten()

            else:

                raise ValueError("Invalid task")

        results[f"{task}_score"] = scores
        results["Prediction"] = (scores > 0.5).astype(int)

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        return results



# HELPERS


    def _prepare_rnn_input(self, X, name):

        if self.scalers[name] is not None:
            X = self.scalers[name].transform(X)

        return X.reshape(X.shape[0], 1, X.shape[1])



        
    def _prepare_dense_input(self, X, name):

        if self.scalers[name] is not None:
            X = self.scalers[name].transform(X)

        return X


# UNLOAD / CLEANUP


    def unload(self):

        global _GLOBAL_MODELS

        try:
            del self.protT5_embedder.model
            del self.protT5_embedder.tokenizer
        except AttributeError:
            pass

        try:
            del self.protAlbert_embedder.model
            del self.protAlbert_embedder.tokenizer
        except AttributeError:
            pass

        # Clear the global ProtAlbert cache too, or the model stays alive
        _GLOBAL_MODELS["protAlbert"] = None
        _GLOBAL_MODELS["protT5"] = None

        for attr in ["abp_model", "afp_model", "ahp_model", "app_model", "avp_model",
                     "acp_model", "acp_scaler", "aip_model"]:
            try:
                delattr(self, attr)
            except AttributeError:
                pass

        try:
            tf.keras.backend.clear_session()
        except Exception:
            pass

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        print("Predictor unloaded, GPU/CPU memory released")
