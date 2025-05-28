from qdrant_client.http.models import PointStruct, CollectionStatus, NamedVector, VectorParams, Distance
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from tqdm import tqdm  # Optional for progress bar
from rel_db_functions import Material
import time

qdrant_client = QdrantClient(host="localhost", port=6333)

COLLECTION_NAME = "Codes"


def create_collection():
    try:
        qdrant_client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "short_desc_ita": VectorParams(size=512, distance=Distance.COSINE),
                "short_desc_eng": VectorParams(size=512, distance=Distance.COSINE),
                "long_desc_ita": VectorParams(size=512, distance=Distance.COSINE),
                "long_desc_eng": VectorParams(size=512, distance=Distance.COSINE),
            }
        )
        print("Collection successufully created")
        return True
    except Exception as e:
        print("Error creating collection: ", e)
        return False


# model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
# model = SentenceTransformer('xlm-r-100langs-bert-base-nli-stsb-mean-tokens')

def create_vector_db():
    materials = Material.query.all()
    batch_size = 100
    point_id = 1  # Auto-incrementing ID counter

    for i in tqdm(range(0, len(materials),batch_size), desc="Processing materials"):
        batch_points = []
        
        for material in materials[i:i+batch_size]:
            # Process texts
            ita_text = ' '.join(filter(None, [
                material.short_desc_it, 
                material.long_desc_it
            ]))
            
            eng_text = ' '.join(filter(None, [
                material.short_desc_eng,
                material.long_desc_eng
            ]))

            # Generate embeddings
            ita_short_embedding = model.encode(material.short_desc_it.lower()).tolist() if material.short_desc_it else []
            eng_short_embedding = model.encode(material.short_desc_eng.lower()).tolist() if material.short_desc_eng else []
            ita_long_embedding = model.encode(ita_text.lower()).tolist() if ita_text else []
            eng_long_embedding = model.encode(eng_text.lower()).tolist() if eng_text else []

            # Create payload with original data
            payload = {
                "material_id": material.id,  # Original ID as string
                "short_desc_it": material.short_desc_it,
                "short_desc_eng": material.short_desc_eng,
                "long_desc_it": material.long_desc_it,
                "long_desc_eng": material.long_desc_eng,
            }

            # Create point with auto-incremented ID
            batch_points.append(PointStruct(
                id=point_id,
                vector={
                    "short_desc_ita": ita_short_embedding,
                    "short_desc_eng": eng_short_embedding,
                    "long_desc_ita": ita_long_embedding,
                    "long_desc_eng": eng_long_embedding
                },
                payload=payload
            ))
            point_id += 1  # Increment for next point

        # Insert batch
        if batch_points:
            qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                points=batch_points
            )

    # Verify collection
    try:
        info = qdrant_client.get_collection(COLLECTION_NAME)
        print(f"Collection ready. Total points: {info.points_count}")
        return True
    except Exception as e:
        print("Verification failed:", e)
        return False


def vector_search(query, top_n=10):
    try:
        # Generate query embedding
        query_embedding = model.encode(query.lower()).tolist()
    except Exception as e:
        print(f"Embedding generation failed: {e}")
        return []

    start_time = time.time()
    
    # Search both vectors simultaneously
    try:
        # Search Italian vectors
        italian_short_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=NamedVector(name="short_desc_ita", vector=query_embedding),
            limit=top_n,
            with_payload=True
        )
        
        # Search English vectors
        english_short_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=NamedVector(name="short_desc_eng", vector=query_embedding),
            limit=top_n,
            with_payload=True
        )

        italian_long_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=NamedVector(name="long_desc_ita", vector=query_embedding),
            limit=top_n,
            with_payload=True
        )

        english_long_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=NamedVector(name="long_desc_eng", vector=query_embedding),
            limit=top_n,
            with_payload=True
        )
    except Exception as e:
        print(f"Search failed: {e}")
        return []

    # Combine and deduplicate results
    combined_results = {}
    for hit in italian_short_results + english_short_results + italian_long_results + english_long_results:
        material_id = hit.payload["material_id"]
        if material_id not in combined_results or hit.score > combined_results[material_id].score:
            combined_results[material_id] = hit

    # Sort by score and select top N
    sorted_results = sorted(combined_results.values(), 
                           key=lambda x: x.score, 
                           reverse=True)[:top_n]

    print(f"Search completed in {time.time() - start_time:.2f} seconds")
    
    return [{
        "material_id": hit.payload["material_id"],
        "score": hit.score,
        "it": hit.payload.get("short_desc_it", ""),
        "en": hit.payload.get("short_desc_eng", ""),
    } for hit in sorted_results]