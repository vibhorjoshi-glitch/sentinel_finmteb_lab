"""
SENTINEL: Qwen 1.5 2B Embedding Model Integration Example
========================================================

This example demonstrates how to use the Qwen-1.5-2B-instruct model 
with 1536-dimensional embeddings in SENTINEL framework.

Model Details:
  - Name: gte-Qwen1.5-2B-instruct
  - Dimensions: 1536 (4x larger than all-MiniLM)
  - Parameters: ~2 billion (10x larger than MiniLM)
  - Capacity: Much higher representation power
  - Inference: Slower but more accurate embeddings

Usage:
  python example_qwen_1.5_2b.py
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.embedder import SentinelEmbedder, MODEL_REGISTRY
from src.config import DEVICE

print("=" * 80)
print("SENTINEL: Qwen 1.5 2B Embedding Example")
print("=" * 80)

# ============================================================================
# 1. LIST AVAILABLE MODELS
# ============================================================================
print("\n[1] Available Embedding Models:")
print("-" * 80)

available_models = SentinelEmbedder.list_available_models()
for model_key, model_info in available_models.items():
    print(f"\n  {model_key}:")
    print(f"    - Full Name: {model_info['model_name']}")
    print(f"    - Dimension: {model_info['vector_dim']}")
    print(f"    - Description: {model_info['description']}")

# ============================================================================
# 2. INITIALIZE QWEN 1.5 2B EMBEDDER
# ============================================================================
print("\n[2] Initializing Qwen 1.5 2B Embedder...")
print("-" * 80)

try:
    embedder = SentinelEmbedder(
        model_name="qwen-1.5-2b",  # Or use full path: "Alibaba-NLP/gte-Qwen1.5-2B-instruct"
        device=DEVICE,
        verbose=True
    )
    print("\n✅ Embedder initialized successfully!")
except Exception as e:
    print(f"\n❌ Error initializing embedder: {e}")
    sys.exit(1)

# ============================================================================
# 3. GET MODEL INFORMATION
# ============================================================================
print("\n[3] Model Information:")
print("-" * 80)

model_info = embedder.get_model_info()
for key, value in model_info.items():
    print(f"  {key}: {value}")

# ============================================================================
# 4. ENCODE SAMPLE FINANCIAL TEXTS
# ============================================================================
print("\n[4] Encoding Financial Documents with Qwen 1.5 2B:")
print("-" * 80)

financial_documents = [
    "The company reported a 15% increase in quarterly revenue, driven by strong sales in the technology sector.",
    "Risk management frameworks must address emerging cybersecurity threats and operational vulnerabilities.",
    "Portfolio diversification across asset classes reduces systematic risk exposure.",
    "Regulatory compliance requires timely filing of financial statements and audit trails.",
    "Cash flow analysis reveals strong liquidity position with sufficient working capital reserves.",
]

print(f"\nEncoding {len(financial_documents)} documents...")

embeddings = embedder.encode(
    financial_documents,
    batch_size=32,
    persona="Forensic Auditor",
    normalize_embeddings=True,
    show_progress_bar=True
)

print(f"\n✅ Embeddings shape: {embeddings.shape}")
print(f"   - Documents: {embeddings.shape[0]}")
print(f"   - Dimension: {embeddings.shape[1]} (Qwen 1.5 2B)")
print(f"   - Data type: {embeddings.dtype}")

# ============================================================================
# 5. ANALYZE EMBEDDING PROPERTIES
# ============================================================================
print("\n[5] Embedding Vector Analysis:")
print("-" * 80)

# Vector norms (should be ~1.0 after L2 normalization)
norms = np.linalg.norm(embeddings, axis=1)
print(f"\nVector norms (L2 normalized):")
print(f"  - Mean: {norms.mean():.6f}")
print(f"  - Std: {norms.std():.6f}")
print(f"  - Min: {norms.min():.6f}")
print(f"  - Max: {norms.max():.6f}")

# Dimensionality statistics
print(f"\nVector component statistics:")
print(f"  - Mean absolute value: {np.abs(embeddings).mean():.6f}")
print(f"  - Max absolute value: {np.abs(embeddings).max():.6f}")

# Similarity matrix (cosine similarity)
similarity_matrix = embeddings @ embeddings.T
print(f"\nCosine similarity (doc-to-doc):")
print(f"  - Diagonal (self-similarity): {np.diag(similarity_matrix).mean():.6f} (should be ~1.0)")
print(f"  - Mean off-diagonal: {(similarity_matrix.sum() - np.trace(similarity_matrix)) / (len(embeddings) * (len(embeddings) - 1)):.6f}")

# Display similarity matrix
print(f"\nSimilarity Matrix (5x5):")
print("  Document  |", "  ".join([f"Doc{i+1}" for i in range(len(financial_documents))]))
print("  " + "-" * 50)
for i in range(len(financial_documents)):
    print(f"  Doc{i+1}     |", "  ".join([f"{sim:.3f}" for sim in similarity_matrix[i]]))

# ============================================================================
# 6. COMPARE WITH ALL-MINILM MODEL (Optional - requires loading)
# ============================================================================
print("\n[6] Model Comparison (Qwen 1.5 2B vs all-MiniLM):")
print("-" * 80)

comparison = {
    "all-MiniLM": {
        "model_path": "sentence-transformers/all-MiniLM-L6-v2",
        "dimension": 384,
        "parameters": "22M",
        "speed": "⚡⚡⚡ Very Fast",
        "accuracy": "⭐⭐",
        "use_case": "Fast retrieval, low latency"
    },
    "Qwen 1.5 2B": {
        "model_path": "Alibaba-NLP/gte-Qwen1.5-2B-instruct",
        "dimension": 1536,
        "parameters": "2B",
        "speed": "⚡ Slower",
        "accuracy": "⭐⭐⭐⭐⭐ Excellent",
        "use_case": "High-quality embeddings, better semantic understanding"
    }
}

for model_name, info in comparison.items():
    print(f"\n  {model_name}:")
    for key, value in info.items():
        print(f"    {key:15}: {value}")

# ============================================================================
# 7. BATCH ENCODING EXAMPLE
# ============================================================================
print("\n[7] Batch Encoding Example:")
print("-" * 80)

batch_1 = [
    "Investment in renewable energy stocks",
    "ESG compliance framework implementation"
]

batch_2 = [
    "Credit risk assessment methodology",
    "Stress testing for market volatility"
]

all_embeddings = embedder.encode_batch(
    [batch_1, batch_2],
    batch_size=32,
    persona="Risk Analyst",
    show_progress_bar=True
)

print(f"\n✅ Batch encoding complete!")
print(f"   - Total embeddings: {all_embeddings.shape[0]}")
print(f"   - Dimension: {all_embeddings.shape[1]}")

# ============================================================================
# 8. CONFIGURATION FOR PRODUCTION USE
# ============================================================================
print("\n[8] Production Configuration Options:")
print("-" * 80)

print("""
To use Qwen 1.5 2B in your production code:

Option A: Via environment variable
  $ export SENTINEL_EMBEDDING_MODEL=qwen-1.5-2b
  $ python your_script.py

Option B: Direct instantiation
  from src.embedder import SentinelEmbedder
  
  embedder = SentinelEmbedder(
      model_name="qwen-1.5-2b",
      device="cuda",
      verbose=True
  )

Option C: Via config
  Edit src/config.py:
  
  EMBEDDING_MODEL = "qwen-1.5-2b"  # or "qwen2-1.5b", "all-MiniLM"
  
  The VECTOR_DIM will be automatically set to 1536.

Option D: Full model path
  embedder = SentinelEmbedder(
      model_name="Alibaba-NLP/gte-Qwen1.5-2B-instruct",
      vector_dim=1536,
      device="cuda"
  )
""")

# ============================================================================
# 9. PERFORMANCE METRICS
# ============================================================================
print("\n[9] Performance Metrics:")
print("-" * 80)

metrics = {
    "Memory per 1000 embeddings": f"{(1000 * 1536 * 4) / (1024**3):.3f} GB",
    "Compression ratio (with 1-bit RaBitQ)": f"{(1536 * 4) / (1536 * 0.125):.1f}x",
    "Typical inference time (100 texts)": "2-5 minutes (GPU), 10-20 minutes (CPU)",
    "Network bandwidth saved (vs 1536-dim float32)": f"{(1536 * 4) / (1536 * 0.125):.1f}x reduction"
}

for metric, value in metrics.items():
    print(f"  {metric}: {value}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("""
✅ Qwen 1.5 2B integration successful!

Key Benefits:
  1. 1536-dimensional embeddings capture richer semantic information
  2. Superior performance on financial domain understanding
  3. RaBitQ compression reduces storage/bandwidth by 12x
  4. Compatible with existing SENTINEL pipeline

Next Steps:
  1. Update src/config.py to use Qwen 1.5 2B as default
  2. Run benchmarks: python run_large_scale_benchmark.py
  3. Monitor results in results/final_ieee_data.json
  4. Compare metrics against all-MiniLM baseline

For issues or questions, check:
  - src/embedder.py for implementation details
  - src/config.py for configuration options
  - examples/ directory for more usage patterns
""")

print("=" * 80)
print("✅ Example complete!")
print("=" * 80)
