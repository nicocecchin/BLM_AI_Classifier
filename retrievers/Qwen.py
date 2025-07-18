from typing import Dict, List, Set, Tuple
from retrievers.Retriever import Retriever
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from retrievers.Item import Item
from tqdm import tqdm
import time
import csv
import torch

class Qwen(Retriever):
    def __init__(self, data_source: str, output_length: int, size: int):
        super().__init__(data_source, output_length)
        
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        self.size = size

        # init model
        if self.size == 1024:
            self.model = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B', device=device)
        elif self.size == 2560:
            self.model = SentenceTransformer('Qwen/Qwen3-Embedding-4B', device=device)
        elif self.size == 4096:
            self.model = SentenceTransformer('Qwen/Qwen3-Embedding-8B', device=device)
        else:
            raise ValueError(f"Unknown model size: {size}")
        
        # create qdrant client and database
        #self.qdrant_client = QdrantClient(":memory:")
        self.qdrant_client = QdrantClient(host="localhost", port=6333)
        if not self.qdrant_client.collection_exists(collection_name="vector-database-"+str(self.size)):
            self.qdrant_client.recreate_collection(
                collection_name="vector-database-"+str(self.size),
                vectors_config={
                    "short_desc_ita": models.VectorParams(size=self.size, distance=models.Distance.COSINE),
                    "short_desc_eng": models.VectorParams(size=self.size, distance=models.Distance.COSINE),
                    "long_desc_ita": models.VectorParams(size=self.size, distance=models.Distance.COSINE),
                    "long_desc_eng": models.VectorParams(size=self.size, distance=models.Distance.COSINE),
                }
            )

            # populate the vector database with items from the catalogue
            points = []
            point_id = 1
            # read CSV and count rows for progress bar
            with open(self.data_source, 'r', encoding='utf-8') as f:
                row_count = sum(1 for _ in f) - 1
            with open(self.data_source, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter=';')
                for row in tqdm(reader, total=row_count, desc="Populating vector database"):
                    # prepare texts
                    short_it = row.get('short_it', '').strip()
                    short_eng = row.get('short_eng', '').strip()
                    long_it = row.get('long_it', '').strip()
                    long_eng = row.get('long_eng', '').strip()

                    # use short description if long is empty
                    if not long_it:
                        long_it = short_it
                    if not long_eng:
                        long_eng = short_eng

                    # in the 1024 model, the input text should start with "query: " or "passage: ", even for non-English texts.
                    # For tasks other than retrieval, you can simply use the "query: " prefix.
                    #if self.size == 1024:
                        #short_it = "passage: " + short_it
                        #short_eng = "passage: " + short_eng
                        #long_it = "passage: " + long_it
                        #long_eng = "passage: " + long_eng

                    # create vectors
                    vector_short_it = self.model.encode(short_it, convert_to_numpy=True)
                    vector_short_eng = self.model.encode(short_eng, convert_to_numpy=True)
                    vector_long_it = self.model.encode(long_it, convert_to_numpy=True)
                    vector_long_eng = self.model.encode(long_eng, convert_to_numpy=True)

                    point = models.PointStruct(
                        id=point_id,
                        vector={
                            "short_desc_ita": vector_short_it,
                            "short_desc_eng": vector_short_eng,
                            "long_desc_ita": vector_long_it,
                            "long_desc_eng": vector_long_eng,
                        },
                        payload={
                            "item_id": row['id'],
                            "short_it": short_it,
                            "short_eng": short_eng,
                            "long_it": long_it,
                            "long_eng": long_eng
                        }
                    )
                    points.append(point)
                    point_id += 1
                    if len(points) >= 100:  # upload points in batches of 100
                        self.qdrant_client.upsert(
                            collection_name="vector-database-"+str(self.size),
                            points=points
                        )
                        points = []
                
            # upload points to the vector database
            if points:
                self.qdrant_client.upsert(
                    collection_name="vector-database-"+str(self.size),
                    points=points
                )
            
        info = self.qdrant_client.get_collection("vector-database-"+str(self.size))
        print(f"Vector database ready. Points: {info.points_count}")

    def retrieve(self, query:str) -> Tuple[List[Tuple[Item, float]], float]:
        start = time.time()

        # encode the query
        query_vector = self.model.encode(query.lower(), convert_to_numpy=True)

        # search in the vector database
        results = self.qdrant_client.search(
            collection_name="vector-database-"+str(self.size),
            query_vector=models.NamedVector(name="long_desc_ita", vector=query_vector),
            limit=self.output_length
        )

        items = []
        for result in results:
            item_id = result.payload["item_id"]
            score = result.score

            item = Item(item_id=item_id, 
                        ita_short_desc=result.payload["short_it"],
                        eng_short_desc=result.payload["short_eng"],
                        ita_long_desc=result.payload["long_it"],
                        eng_long_desc=result.payload["long_eng"])
            items.append((item, score))

        retrieve_time = time.time() - start
        return items, retrieve_time