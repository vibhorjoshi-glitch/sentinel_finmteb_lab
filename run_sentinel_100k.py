import time
from datasets import load_dataset
from src.embedder import SentinelEmbedder
from src.engine import SentinelEngine
from qdrant_client import models

def main():
    # 1. Load Data
    print("--- Loading Financial Documents ---")
    ds = load_dataset("mteb/fiqa", "corpus", split="corpus")
    
    # FiQA corpus typically has 57K docs, so we use what's available
    texts = [f"{row['title']} {row['text']}" for row in ds]
    ids = list(range(len(texts)))
    
    print(f"Loaded {len(texts)} documents from FiQA corpus")

    # 2. Setup
    embedder = SentinelEmbedder()
    engine = SentinelEngine()
    engine.init_collection()

    try:
        # Phase 1 & 2: Vectorization & Ingestion
        print(f"Vectorizing {len(texts)} docs (Rotated + Persona-Aware)...")
        start_vec = time.time()
        vectors = embedder.encode(texts)
        vec_time = time.time() - start_vec
        print(f"Vectorization done in {vec_time:.1f}s ({len(texts)/vec_time:.1f} docs/sec)")

        print("Ingesting into 32x Compressed Manifold...")
        start_ingest = time.time()
        engine.client.upsert(
            collection_name="sentinel_100k_manifold",
            points=models.Batch(ids=ids, vectors=vectors.tolist())
        )
        ingest_time = time.time() - start_ingest
        print(f"Ingestion done in {ingest_time:.1f}s")

        # Phase 3: Networking Benchmark (The 160Gbps -> 5Gbps proof)
        # Storage calculation: N docs × 1536 dims × bits_per_dim
        f32_storage_mb = len(texts) * 1536 * 4 / (1024 * 1024)  # Float32 = 4 bytes
        binary_storage_mb = len(texts) * 1536 * 1 / (1024 * 1024 * 8)  # 1-bit = 0.125 bytes
        
        reduction = f32_storage_mb / binary_storage_mb
        
        print(f"\n--- IEEE TMLCN RESEARCH RESULTS ---")
        print(f"Dataset size: {len(texts)} documents")
        print(f"Storage (f32): {f32_storage_mb:.1f} MB")
        print(f"Storage (1-bit): {binary_storage_mb:.1f} MB")
        print(f"Compression Ratio: {reduction:.1f}x")
        print(f"Simulated Network Load: {(160/reduction):.2f} Gbps (Mitigated from 160 Gbps)")
        print(f"\n✅ Research pipeline complete!")

    finally:
        # FIX: Manual close prevents the QdrantClient.__del__ error
        print("Closing Sovereign Engine Safely...")
        engine.close()
        print("✓ Engine closed without errors")

if __name__ == "__main__":
    main()

