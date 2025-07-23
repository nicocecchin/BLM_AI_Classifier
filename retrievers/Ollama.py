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
                    "desc_ita": models.VectorParams(size=self.size, distance=models.Distance.COSINE),
                    "desc_eng": models.VectorParams(size=self.size, distance=models.Distance.COSINE),
                }
            )

            # retrive all items from the catalogue
            catalogue_items_ids:List[int] = []
            catalogue_items_short_ita:List[str] = []
            catalogue_items_short_eng:List[str] = []
            catalogue_items_long_ita:List[str] = []
            catalogue_items_long_eng:List[str] = []
            catalogue_items_ita:List[str] = []
            catalogue_items_eng:List[str] = []
            with open(self.data_source, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter=',')
                for row in reader:
                    # prepare texts
                    short_ita = row.get('short_ita', '').strip()
                    short_eng = row.get('short_eng', '').strip()
                    long_ita = row.get('long_ita', '').strip()
                    long_eng = row.get('long_eng', '').strip()

                    ita = "corta: " + short_ita 
                    if long_ita:
                        ita += f" | lunga: {long_ita}"
                    eng = "short: " + short_eng
                    if long_eng:
                        eng += f" | long: {long_eng}"

                    # use short description if long is empty
                    if not long_ita:
                        long_ita = short_ita
                    if not long_eng:
                        long_eng = short_eng

                    catalogue_items_short_ita.append(short_ita)
                    catalogue_items_short_eng.append(short_eng)
                    catalogue_items_long_ita.append(long_ita)
                    catalogue_items_long_eng.append(long_eng)
                    catalogue_items_ids.append(row['id'])
                    catalogue_items_ita.append(ita)
                    catalogue_items_eng.append(eng)
                    

            # create vector embeddings
            catalogue_items_ita_vectors = self.model.get_text_embedding_batch(catalogue_items_ita, show_progress=True)
            catalogue_items_eng_vectors = self.model.get_text_embedding_batch(catalogue_items_eng, show_progress=True)

            assert len(catalogue_items_ita_vectors) == len(catalogue_items_eng_vectors), "Vectors length mismatch"
            assert len(catalogue_items_ita_vectors) == len(catalogue_items_ita) == len(catalogue_items_ids), "Vectors and items length mismatch"

            # populate the vector database with items from the catalogue
            points = []
            point_id = 1
            # read CSV and count rows for progress bar
            for i in tqdm(range(len(catalogue_items_short_ita)), desc="Populating vector database"):
                point = models.PointStruct(
                    id=point_id,
                    vector={
                        "desc_ita": catalogue_items_ita_vectors[i],
                        "desc_eng": catalogue_items_eng_vectors[i],
                    },
                    payload={
                        "item_id": catalogue_items_ids[i],
                        "short_ita": catalogue_items_short_ita[i],
                        "short_eng": catalogue_items_short_eng[i],
                        "long_ita": catalogue_items_long_ita[i],
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

    def retrieve(self, query:str, language: str = None) -> Tuple[List[Tuple[Item, float]], float]:
        start = time.time()
        super().retrieve(query, language=language)

        # encode the query
        query_vector = self.model.get_query_embedding(query.lower())

        # search in the vector database
        results = self.qdrant_client.search(
            collection_name="vector-database-lama",
            query_vector=models.NamedVector(name=f"desc_{self.language}", vector=query_vector),
            limit=self.output_length
        )

        items = []
        for result in results:
            item_id = result.payload["item_id"]
            score = result.score

            item = Item(item_id=item_id, 
                        ita_short_desc=result.payload["short_ita"],
                        eng_short_desc=result.payload["short_eng"],
                        ita_long_desc=result.payload["long_ita"],
                        eng_long_desc=result.payload["long_eng"])
            items.append((item, score))

        retrieve_time = time.time() - start
        return items, retrieve_time