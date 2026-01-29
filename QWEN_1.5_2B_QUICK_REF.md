# Quick Reference: Qwen 1.5 2B in SENTINEL

## TL;DR - Just Get Started

### 1. Load Qwen 1.5 2B Model

```python
from src.embedder import SentinelEmbedder

# Initialize with Qwen 1.5 2B (1536 dimensions)
embedder = SentinelEmbedder("qwen-1.5-2b")

# Or use full model path
embedder = SentinelEmbedder("Alibaba-NLP/gte-Qwen1.5-2B-instruct")
```

### 2. Encode Financial Documents

```python
docs = [
    "Investment risk analysis for portfolio diversification",
    "Quarterly earnings report shows 15% revenue growth",
    "Regulatory compliance framework update"
]

# Encode with financial persona
embeddings = embedder.encode(
    docs,
    persona="Risk Analyst",  # or "Forensic Auditor", "CFO", etc.
    normalize_embeddings=True
)

# Result: (3, 1536) numpy array with L2-normalized vectors
print(f"Shape: {embeddings.shape}")
print(f"Dtype: {embeddings.dtype}")
```

### 3. Use in Benchmark

```bash
# Method A: Via environment variable
export SENTINEL_EMBEDDING_MODEL=qwen-1.5-2b
python run_large_scale_benchmark.py

# Method B: Edit config.py
EMBEDDING_MODEL = "qwen-1.5-2b"
python run_large_scale_benchmark.py
```

### 4. Check Model Info

```python
# List all available models
models = SentinelEmbedder.list_available_models()
for name, config in models.items():
    print(f"{name}: {config['description']}")

# Get current model info
info = embedder.get_model_info()
print(f"Vector dimension: {info['vector_dim']}")
print(f"Device: {info['device']}")
```

---

## Model Comparison

| Feature | all-MiniLM | Qwen 1.5 2B |
|---------|-----------|-----------|
| Dimension | 384 | 1536 |
| Parameters | 22M | 2B |
| Size | ~138 MB | ~4.5 GB |
| Speed | Very Fast | Slower |
| Quality | Good | Excellent |
| Best For | Real-time | Accuracy |

---

## Code Snippets

### Basic Usage
```python
from src.embedder import SentinelEmbedder

embedder = SentinelEmbedder("qwen-1.5-2b")
vectors = embedder.encode(["text1", "text2"])
```

### With Custom Persona
```python
embedder = SentinelEmbedder("qwen-1.5-2b", device="cuda")

vectors = embedder.encode(
    texts,
    persona="Portfolio Manager",
    batch_size=32,
    show_progress_bar=True
)
```

### Batch Processing
```python
batch_1 = ["Risk analysis document"]
batch_2 = ["Compliance report"]

all_vectors = embedder.encode_batch(
    [batch_1, batch_2],
    persona="Compliance Officer"
)
```

### Compute Similarity
```python
import numpy as np

# Cosine similarity
similarity = embeddings @ embeddings.T

# Find most similar documents
query_vector = embeddings[0]
similarities = query_vector @ embeddings.T
top_k = np.argsort(similarities)[::-1][:5]
```

---

## Configuration

### Environment Variables
```bash
# Set embedding model
export SENTINEL_EMBEDDING_MODEL=qwen-1.5-2b

# Enable reranking
export SENTINEL_ENABLE_RERANKING=True

# Set reranking model
export SENTINEL_RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

### Python Config
```python
# src/config.py
EMBEDDING_MODEL = "qwen-1.5-2b"  # Automatically sets VECTOR_DIM = 1536
ENABLE_RERANKING = True
RERANK_TOP_K = 50
```

---

## Supported Models

```python
# All built-in models
{
    "all-MiniLM": 384,           # Fast, lightweight
    "qwen-1.5-2b": 1536,         # Large capacity, high quality
    "qwen2-1.5b": 1536           # Latest Qwen version
}

# Or use custom model path
SentinelEmbedder("Alibaba-NLP/gte-Qwen1.5-2B-instruct", vector_dim=1536)
```

---

## Performance Tips

### Speed Up Encoding
```python
# Increase batch size (if memory allows)
embeddings = embedder.encode(texts, batch_size=128)

# Use GPU
embedder = SentinelEmbedder("qwen-1.5-2b", device="cuda")

# Disable progress bar
embeddings = embedder.encode(texts, show_progress_bar=False)
```

### Save Memory
```python
# Reduce batch size
embeddings = embedder.encode(texts, batch_size=16)

# Convert to lower precision if needed
embeddings = embeddings.astype(np.float16)  # Half precision

# Use CPU for very large batches
embedder = SentinelEmbedder("qwen-1.5-2b", device="cpu")
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Model download fails | Pre-download: `SentenceTransformer("Alibaba-NLP/gte-Qwen1.5-2B-instruct")` |
| Out of memory | Reduce batch_size or use GPU |
| Slow on CPU | Switch to GPU: `device="cuda"` |
| Shape mismatch | Check `embedder.vector_dim` (should be 1536) |
| Cache error | Already fixed in updated embedder.py |

---

## Run Example

```bash
python example_qwen_1.5_2b.py
```

This runs a complete example showing:
- Model listing
- Initialization
- Encoding
- Similarity analysis
- Batch processing
- Performance metrics

---

## Integration Files

| File | Purpose |
|------|---------|
| `src/embedder.py` | Core embedder with model registry |
| `src/config.py` | Configuration with model selection |
| `example_qwen_1.5_2b.py` | Complete working example |
| `QWEN_1.5_2B_INTEGRATION.md` | Full integration guide |
| `QWEN_1.5_2B_QUICK_REF.md` | This quick reference |

---

## Key Metrics

### Vector Properties
- **Dimension**: 1536 (4x more than all-MiniLM)
- **Compression**: 12x reduction with RaBitQ
- **Normalized**: L2 normalized (unit vectors)
- **Device**: Auto-detect (CUDA/CPU)

### Benchmark Impact
- **Recall@10**: ~+10% improvement expected
- **NDCG**: ~+15% improvement expected
- **Inference**: 3-5x slower (quality trade-off)

---

## Resources

- **HuggingFace Model Card**: [Alibaba-NLP/gte-Qwen1.5-2B-instruct](https://huggingface.co/Alibaba-NLP/gte-Qwen1.5-2B-instruct)
- **GitHub**: [QwenLM/Qwen](https://github.com/QwenLM/Qwen)
- **Documentation**: Run `python example_qwen_1.5_2b.py`

---

## Next Steps

1. ✅ Run example: `python example_qwen_1.5_2b.py`
2. ✅ Update config: Set `EMBEDDING_MODEL = "qwen-1.5-2b"`
3. ✅ Run benchmark: `python run_large_scale_benchmark.py`
4. ✅ Check results: View `results/final_ieee_data.json`
5. ✅ Compare metrics with baseline (all-MiniLM)

---

**Last Updated**: January 29, 2026 | **Status**: ✅ Ready to Use
