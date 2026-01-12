import time
import os
import torch
from datasets import load_dataset
from src.embedder import SentinelEmbedder
from src.engine import SentinelEngine
from qdrant_client import models

def run_test(sample_size=100):
    print(f"--- 🔍 Starting Sentinel Validation ({sample_size} docs) ---")
    
    # 1. Load Data
    try:
        ds = load_dataset('mteb/fiqa', 'corpus', split='corpus')
        texts = [f"{row['title']} {row['text']}" for row in ds][:sample_size]
        ids = list(range(len(texts)))
    except Exception as e:
        print(f"❌ Dataset load failed: {e}")
        return

    # 2. Initialize
    embedder = SentinelEmbedder()
    engine = SentinelEngine()
    engine.init_collection()

    try:
        print(f"Vectorizing {len(texts)} docs (Qwen-2.5 + RaBitQ Rotation)...")
        start = time.time()
        vectors = embedder.encode(texts)
        duration = time.time() - start
        print(f"✅ Vectorization done in {duration:.2f}s ({(duration/len(texts)):.4f}s per doc)")
        
        print('Ingesting into 32x Compressed Manifold...')
        engine.client.upsert(
            collection_name='sentinel_100k_manifold',
            points=models.Batch(ids=ids, vectors=vectors.tolist())
        )
        print('✅ Ingestion successful')
        
        # 3. IEEE Metrics Calculation
        # f32 = 4 bytes per dim | 1-bit = 1 bit per dim
        f32_bytes = len(texts) * 1536 * 4
        binary_bytes = len(texts) * 1536 / 8
        
        f32_mb = f32_bytes / (1024 * 1024)
        binary_mb = binary_bytes / (1024 * 1024)
        ratio = f32_mb / binary_mb
        
        print(f'\n--- 📊 TEST RESULTS (IEEE TMLCN) ---')
        print(f'Full Precision (f32) Storage: {f32_mb:.4f} MB')
        print(f'Sovereign (1-bit) Storage:    {binary_mb:.4f} MB')
        print(f'Theoretical Compression:      {ratio:.1f}x')
        print(f'Estimated 100K Vectorization: ~9.5 hours on CPU')
        
    finally:
        engine.close()
        print('\n✓ Sovereign Engine closed safely.')

if __name__ == "__main__":
    run_test(1000) 
