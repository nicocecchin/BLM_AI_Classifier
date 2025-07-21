from typing import Dict, List, Set, Tuple
from retrievers.Retriever import Retriever
from qdrant_client import QdrantClient, models
from retrievers.Item import Item
from tqdm import tqdm
from llama_index.embeddings.ollama import OllamaEmbedding
import time
import csv
import torch
import ollama

class Ollama(Retriever):
    def __init__(self, data_source: str, output_length: int):
        super().__init__(data_source, output_length)
        
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        self.model = OllamaEmbedding(model_name="llama3", device=device, base_url="http://localhost:11434", ollama_additional_kwargs={"microstat": 0}, )
        
        # create qdrant client and database
        #self.qdrant_client = QdrantClient(":memory:")
        self.qdrant_client = QdrantClient(host="localhost", port=6333, timeout=60.0)
        if not self.qdrant_client.collection_exists(collection_name="vector-database-lama"):
            self.qdrant_client.recreate_collection(
                collection_name="vector-database-lama",
                vectors_config={
                    "short_desc_ita": models.VectorParams(size=4096, distance=models.Distance.COSINE),
                    "short_desc_eng": models.VectorParams(size=4096, distance=models.Distance.COSINE),
                    "long_desc_ita": models.VectorParams(size=4096, distance=models.Distance.COSINE),
                    "long_desc_eng": models.VectorParams(size=4096, distance=models.Distance.COSINE),
                }
            )

            # retrive all items from the catalogue
            catalogue_items_ids:List[int] = []
            catalogue_items_short_it:List[str] = []
            catalogue_items_short_eng:List[str] = []
            catalogue_items_long_it:List[str] = []
            catalogue_items_long_eng:List[str] = []
            with open(self.data_source, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter=';')
                for row in reader:
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

                    catalogue_items_short_it.append(short_it)
                    catalogue_items_short_eng.append(short_eng)
                    catalogue_items_long_it.append(long_it)
                    catalogue_items_long_eng.append(long_eng)
                    catalogue_items_ids.append(row['id'])
                    

            # create vector embeddings
            catalogue_items_short_it_vectors = self.model.get_text_embedding_batch(catalogue_items_short_it, show_progress=True)
            catalogue_items_short_eng_vectors = self.model.get_text_embedding_batch(catalogue_items_short_eng, show_progress=True)
            catalogue_items_long_it_vectors = self.model.get_text_embedding_batch(catalogue_items_long_it, show_progress=True)
            catalogue_items_long_eng_vectors = self.model.get_text_embedding_batch(catalogue_items_long_eng, show_progress=True)

            assert len(catalogue_items_short_it_vectors) == len(catalogue_items_short_eng_vectors) == len(catalogue_items_long_it_vectors) == len(catalogue_items_long_eng_vectors), "Vectors length mismatch"
            assert len(catalogue_items_short_it_vectors) == len(catalogue_items_short_it) == len(catalogue_items_ids), "Vectors and items length mismatch"

            # populate the vector database with items from the catalogue
            points = []
            point_id = 1
            # read CSV and count rows for progress bar
            for i in tqdm(range(len(catalogue_items_short_it)), desc="Populating vector database"):
                point = models.PointStruct(
                    id=point_id,
                    vector={
                        "short_desc_ita": catalogue_items_short_it_vectors[i],
                        "short_desc_eng": catalogue_items_short_eng_vectors[i],
                        "long_desc_ita": catalogue_items_long_it_vectors[i],
                        "long_desc_eng": catalogue_items_long_eng_vectors[i],
                    },
                    payload={
                        "item_id": catalogue_items_ids[i],
                        "short_it": catalogue_items_short_it[i],
                        "short_eng": catalogue_items_short_eng[i],
                        "long_it": catalogue_items_long_it[i],
                        "long_eng": catalogue_items_long_eng[i]
                    }
                )
                points.append(point)
                point_id += 1
                if len(points) >= 20:  # upload points in batches of 20
                    self.qdrant_client.upsert(
                        collection_name="vector-database-lama",
                        points=points
                    )
                    points = []
                
            # upload points to the vector database
            if points:
                self.qdrant_client.upsert(
                    collection_name="vector-database-lama",
                    points=points
                )
            
        info = self.qdrant_client.get_collection("vector-database-lama")
        print(f"Vector database ready. Points: {info.points_count}")

    def retrieve(self, query:str) -> Tuple[List[Tuple[Item, float]], float]:
        start = time.time()

        # encode the query
        query_vector = self.model.get_query_embedding(query.lower())

        # search in the vector database
        results = self.qdrant_client.search(
            collection_name="vector-database-lama",
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