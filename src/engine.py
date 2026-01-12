from qdrant_client import QdrantClient, models
from .config import DATA_PATH, COLLECTION_NAME, VECTOR_DIM

class SentinelEngine:
    def __init__(self):
        self.client = QdrantClient(path=DATA_PATH)

    def init_collection(self):
        if not self.client.collection_exists(COLLECTION_NAME):
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=VECTOR_DIM, 
                    distance=models.Distance.COSINE,
                    on_disk=True # Essential for 100K scale
                ),
                quantization_config=models.BinaryQuantization(
                    binary=models.BinaryQuantizationConfig(always_ram=True)
                )
            )

    def close(self):
        """Explicitly close the client to prevent __del__ exceptions"""
        self.client.close()
