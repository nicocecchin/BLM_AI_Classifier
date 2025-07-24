import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from retrievers.Fuzzy import Fuzzy
from rankers.FuzzyRanker import FuzzyRanker
from rankers.Bm25Ranker import Bm25Ranker
from retrievers.Bm25 import Bm25
from retrievers.Sbert import Sbert
from retrievers.Retriever import Retriever
from typing import List, Tuple
from retrievers.Item import Item
# from retrievers.Ollama import Ollama
from rankers.CrossEncoderModel import CrossEncoderModel
from rankers.BGEReranker import BGEReranker
import time

import pandas as pd

class HybridRetriever(Retriever):
    def __init__(self, data_source: str, output_length: int, retriever_name:str, ranker_name:str, model_name:str = None):
        super().__init__(data_source, output_length)
        self.retriever_length = 100
        if retriever_name == "bm25":
            self.retriever = Bm25(data_source, self.retriever_length)
        elif retriever_name == "fuzzy":
            self.retriever = Fuzzy(data_source, self.retriever_length, method='token_set_ratio')
        # elif retriever_name == "Ollama":
        #     self.retriever = Ollama(data_source, self.retriever_length)
        elif retriever_name == "sbert_1024":
            self.retriever = Sbert(data_source, self.retriever_length, size=1024)
        else:
            raise ValueError(f"Unknown retriever: {retriever_name}")

        self.output_length = output_length
        self.ranker_name = ranker_name
        if ranker_name == "cross_encoder":
            if model_name is None:
                raise ValueError("Model name must be provided for CrossEncoder ranker")
            self.ranker = CrossEncoderModel(self.output_length, model_name=model_name)
        elif ranker_name == "fuzzy":
            if model_name is None:
                raise ValueError("Method must be provided for Fuzzy ranker")
            self.ranker = FuzzyRanker(self.output_length, method=model_name)
        elif ranker_name == "bm25":
            self.ranker = Bm25Ranker(self.output_length)
        elif ranker_name == "bge_m3":
            self.ranker = BGEReranker(self.output_length, model_name=ranker_name)
        elif ranker_name == "bge_gemma":
            self.ranker = BGEReranker(self.output_length, model_name=ranker_name)
        else:
            raise ValueError(f"Unknown ranker: {ranker_name}")


    def retrieve(self, query, language: str = None) -> Tuple[List[Tuple[Item, float]], float]:
        start = time.time()

        documents, _ = self.retriever.retrieve(query, language=language)

        if not documents:
            return [], time.time() - start

        ranked_documents = self.ranker.rank(documents, query, language=language)

        return ranked_documents[0:10], time.time() - start


