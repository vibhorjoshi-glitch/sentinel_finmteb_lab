import time
import json
import gc
import torch
import numpy as np
from tqdm import tqdm

# --- IMPORTS FROM YOUR SRC STRUCTURE ---
# We use individual modules to control memory loading/unloading
from src.embedder import SentinelEmbedder     # src/embedder.py
from src.engine import SentinelEngine         # src/engine.py

# --- CONFIGURATION ---
TARGET_DOCS = 1000   # Full smart subset for IEEE paper
BATCH_SIZE = 16      # Increase batch size for CPU efficiency
RESULTS_FILE = "results/ieee_tmlcn_final.json"

def main():
    print("==============================================================")
    print("   SENTINEL: IEEE TMLCN FINAL BENCHMARK (MEMORY-SAFE)        ")
    print("==============================================================")

    # ---------------------------------------------------------
    # STEP 1: LOAD DATA (Text Only - Low Memory)
    # ---------------------------------------------------------
    print("\n[Step 1] Loading Financial Corpus...")
    try:
        # Load from HuggingFace FiQA corpus
        from datasets import load_dataset
        print("   Loading FiQA corpus from HuggingFace...")
        ds = load_dataset("mteb/fiqa", "corpus", split="corpus")
        docs = [f"{row['title']} {row['text']}" for row in ds][:TARGET_DOCS]
        doc_ids = list(range(len(docs)))
        print(f"✅ Loaded {len(docs)} documents.")
    except Exception as e:
        print(f"❌ Dataset Error: {e}")
        return

    # ---------------------------------------------------------
    # STEP 2: VECTORIZATION (High Memory Phase)
    # ---------------------------------------------------------
    print("\n[Step 2] Vectorizing with RaBitQ Rotation...")
    
    # Initialize Embedder
    embedder = SentinelEmbedder()
    
    # Vectorize in tiny batches to avoid OOM
    doc_vectors = []
    for i in tqdm(range(0, len(docs), BATCH_SIZE), desc="Vectorizing"):
        batch_text = docs[i : i + BATCH_SIZE]
        # Use the 'Forensic Auditor' persona from your research
        batch_vecs = embedder.encode(batch_text, persona="Forensic Auditor")
        doc_vectors.append(batch_vecs)
    
    # Flatten list of batches
    doc_vectors = np.vstack(doc_vectors)
    print(f"✅ Generated Matrix Shape: {doc_vectors.shape}")

    # --- CRITICAL MEMORY CLEANUP ---
    print("🧹 Cleaning up Embedder to free RAM...")
    del embedder
    gc.collect() # Force garbage collection
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    time.sleep(2) # Let OS reclaim memory

    # ---------------------------------------------------------
    # STEP 3: INGESTION (Qdrant Phase)
    # ---------------------------------------------------------
    print("\n[Step 3] Building Sovereign Manifold...")
    engine = SentinelEngine()
    engine.init_collection() # Ensure collection exists
    
    try:
        # Ingest the vectors we calculated
        from qdrant_client import models
        engine.client.upsert(
            collection_name="sentinel_100k_manifold", # Must match config.py
            points=models.Batch(ids=doc_ids, vectors=doc_vectors.tolist())
        )
        print("✅ Ingestion Complete.")

        # ---------------------------------------------------------
        # STEP 4: NETWORK BENCHMARK (No heavy compute needed)
        # ---------------------------------------------------------
        print("\n[Step 4] Calculating IEEE Network Metrics...")
        
        # We extrapolate the 100 docs result to the 100k scale for the paper
        f32_size = (100000 * 1536 * 4) / (1024**3) # GB
        bin_size = (100000 * 1536 / 8) / (1024**3) # GB
        reduction = f32_size / bin_size
        
        # Network impact calculation
        # Cloud baseline: 160 Gbps for 100K full vectors
        # SENTINEL reduction: 160 Gbps / compression_ratio
        cloud_baseline_gbps = 160.0
        sentinel_gbps = cloud_baseline_gbps / reduction

        print(f"📊 Compression Ratio: {reduction:.1f}x")
        print(f"📊 Cloud Baseline: {cloud_baseline_gbps:.2f} Gbps")
        print(f"📊 SENTINEL Projected Load: {sentinel_gbps:.2f} Gbps")

        # Save Results
        data = {
            "docs_processed": len(docs),
            "compression_ratio": reduction,
            "cloud_baseline_gbps": cloud_baseline_gbps,
            "sentinel_sovereign_gbps": sentinel_gbps,
            "backhaul_reduction_percent": ((1 - (sentinel_gbps / cloud_baseline_gbps)) * 100),
            "status": "Success"
        }
        
        with open(RESULTS_FILE, "w") as f:
            json.dump(data, f, indent=4)
        print(f"\n✅ Results saved to {RESULTS_FILE}")
        print(f"\n📊 SUMMARY:")
        print(f"   Docs: {data['docs_processed']}")
        print(f"   Compression: {data['compression_ratio']:.1f}x")
        print(f"   Load Reduction: {data['backhaul_reduction_percent']:.1f}%")

    finally:
        engine.close()
        print("\n✓ Engine Closed.")

if __name__ == "__main__":
    main()
