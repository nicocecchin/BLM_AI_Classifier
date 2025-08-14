from typing import Dict, List, Tuple
from retrievers.Retriever import Retriever
from retrievers.Item import Item
from rank_bm25 import BM25Okapi
import time
import csv
import re

class Bm25Matrix(Retriever):
    def __init__(self, data_source: str, output_length: int):
        super().__init__(data_source, output_length)

        self.data_source = data_source
        self.items: Dict[str, Item] = {}
        self.item_list: List[Item] = []           # preserve index order for BM25 scores
        self.catalog_ita_raw: List[str] = []      # raw normalized strings
        self.catalog_eng_raw: List[str] = []
        self.catalog_ita_tok: List[List[str]] = []  # tokenized docs for BM25
        self.catalog_eng_tok: List[List[str]] = []

        # read CSV and populate items + catalogs
        with open(self.data_source, 'r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file, delimiter=',')
            next(reader, None)  # skip header if present
            for row in reader:
                # expects same CSV layout as your original code
                item_id = row[0]
                short_desc_ita = row[1] if len(row) > 1 else ''
                short_desc_eng = row[2] if len(row) > 2 else ''
                long_desc_ita = row[3] if len(row) > 3 else ''
                long_desc_eng = row[4] if len(row) > 4 else ''

                item = Item(
                    item_id=item_id,
                    ita_short_desc=short_desc_ita,
                    eng_short_desc=short_desc_eng,
                    ita_long_desc=long_desc_ita,
                    eng_long_desc=long_desc_eng
                )
                self.items[item_id] = item
                self.item_list.append(item)

                # normalize and tokenize
                text_ita = self.normalize_text(f"{short_desc_ita} {long_desc_ita or ''}")
                text_eng = self.normalize_text(f"{short_desc_eng} {long_desc_eng or ''}")
                self.catalog_ita_raw.append(text_ita)
                self.catalog_eng_raw.append(text_eng)
                self.catalog_ita_tok.append(self.tokenize(text_ita))
                self.catalog_eng_tok.append(self.tokenize(text_eng))

        # build BM25 models for both languages
        # if either catalog is empty, handle gracefully by creating an empty list
        self.bm25_ita = BM25Okapi(self.catalog_ita_tok) if len(self.catalog_ita_tok) > 0 else None
        self.bm25_eng = BM25Okapi(self.catalog_eng_tok) if len(self.catalog_eng_tok) > 0 else None

    def normalize_text(self, text: str) -> str:
        """Lowercase and remove punctuation (keeps word characters and whitespace)."""
        if not text:
            return ''
        text = re.sub(r'[^\w\s]', ' ', text)  # replace punctuation with space to avoid word joins
        return text.lower().strip()

    def tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenizer. You can replace with a more advanced tokenizer if desired."""
        if not text:
            return []
        # split on whitespace, drop empty tokens
        return [t for t in re.split(r'\s+', text) if t]

    def retrieve(self, query: str, language: str = None) -> Tuple[List[Tuple[Item, float]], float]:
        """
        Returns a list of (Item, score) sorted by BM25 score desc and the elapsed time in seconds.
        """
        start = time.time()
        super().retrieve(query, language=language)  # keeps same behavior as your original

        query_normalized = self.normalize_text(query)
        query_tokens = self.tokenize(query_normalized)

        if self.language == 'ita':
            if self.bm25_ita is None:
                scores = [0.0] * len(self.item_list)
            else:
                scores = self.bm25_ita.get_scores(query_tokens)
        elif self.language == 'eng':
            if self.bm25_eng is None:
                scores = [0.0] * len(self.item_list)
            else:
                scores = self.bm25_eng.get_scores(query_tokens)
        else:
            raise ValueError(f"Unknown language: {self.language}")

        # pair each item with its score (preserving the same index order used to build BM25)
        results = [(self.item_list[idx], float(scores[idx])) for idx in range(len(self.item_list))]

        # sort and limit
        results.sort(key=lambda x: x[1], reverse=True)
        results = results[:self.output_length]

        end = time.time()
        return results, end - start
