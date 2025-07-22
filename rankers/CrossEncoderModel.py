from typing import List, Tuple
import time
import csv
import torch
from sentence_transformers import CrossEncoder
from retrievers.Item import Item
from rankers.Ranker import Ranker


class CrossEncoderModel(Ranker):
    def __init__(self, output_length: int, model_name: str):
        super().__init__(output_length)        
        # Initialize the cross-encoder model
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        if model_name == "ms-marco-MiniLM-L-6-v2":
            self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)
        elif model_name == "ms-marco-TinyBERT-L-2-v2":
            self.model = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2", device=device)
        else:
            raise ValueError(f"Unknown model name: {model_name}. Supported models are 'cross-encoder/ms-marco-MiniLM-L-6-v2' and 'cross-encoder/ms-marco-TinyBERT-L-2-v2'.")

    def rank(self, documents: List[Tuple[Item, float]], query: str) -> List[Tuple[Item, float]]:
        """
        Rank documents based on the query using the cross-encoder model.
        """
        # Extract the document texts and their corresponding scores
        results = []
        for item, _ in documents:
            desc_it = f"corta: {item.ita_short_desc}"
            if item.ita_long_desc:
                desc_it += f" | lunga: {item.ita_long_desc}"
            desc_eng = f"short: {item.eng_short_desc}"
            if item.eng_long_desc:
                desc_eng += f" | long: {item.eng_long_desc}"

            # Compare query with Italian and English descriptions
            score_it = self.model.predict([(query, desc_it)])[0]
            score_eng = self.model.predict([(query, desc_eng)])[0]
            if score_it >= score_eng:
                results.append((item, score_it))
            else:
                results.append((item, score_eng))

        # Sort results by score in descending order
        results.sort(key=lambda x: x[1], reverse=True)
        return results