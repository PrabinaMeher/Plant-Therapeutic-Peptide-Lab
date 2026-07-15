import torch
import numpy as np
from transformers import T5EncoderModel, T5Tokenizer


class ProtT5Embedder:

    def __init__(self):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Loading ProtT5 model...")

        self.tokenizer = T5Tokenizer.from_pretrained(
            "Rostlab/prot_t5_xl_uniref50",
            do_lower_case=False
        )

        self.model = T5EncoderModel.from_pretrained(
            "Rostlab/prot_t5_xl_uniref50"
        )

        self.model = self.model.to(self.device)
        self.model.eval()

        # Memory optimization
        if self.device.type == "cuda":
            self.model = self.model.half()

        torch.set_num_threads(2)

        print("ProtT5 loaded successfully")

    def embed(self, sequences, batch_size=1):

        embeddings = []

        for i in range(0, len(sequences), batch_size):

            batch = sequences[i:i + batch_size]
            batch = [" ".join(list(seq)) for seq in batch]

            tokens = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True
            )

            tokens = {k: v.to(self.device) for k, v in tokens.items()}

            with torch.no_grad():
                outputs = self.model(**tokens)

            # Mean pooling
            emb = outputs.last_hidden_state.mean(dim=1)

            embeddings.append(emb.cpu().numpy())

            del tokens
            del outputs
            torch.cuda.empty_cache()

        return np.vstack(embeddings)
