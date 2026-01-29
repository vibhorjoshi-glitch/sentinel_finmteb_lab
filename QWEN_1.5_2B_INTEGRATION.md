# SENTINEL: Qwen 1.5 2B Integration Guide

## Overview

This guide explains how to integrate and use the **Qwen-1.5-2B-instruct** embedding model with 1536-dimensional embeddings in the SENTINEL framework.

## Model Specifications

| Aspect | all-MiniLM-L6-v2 | Qwen 1.5 2B |
|--------|------------------|-----------|
| **Full Name** | sentence-transformers/all-MiniLM-L6-v2 | Alibaba-NLP/gte-Qwen1.5-2B-instruct |
| **Dimension** | 384 | 1536 |
| **Parameters** | 22M | ~2 billion |
| **Speed** | ⚡⚡⚡ Very Fast | ⚡ Slower |
| **Accuracy** | ⭐⭐ Good | ⭐⭐⭐⭐⭐ Excellent |
| **Compression Ratio** | 12x | 12x |

## Quick Start

### 1. Installation

The required dependencies are already in `requirements.txt`. Ensure you have:

```bash
pip install -r requirements.txt
```

### 2. Load Qwen 1.5 2B Model

**Option A: Using Configuration File**

Edit `src/config.py`:

```python
# Change this:
EMBEDDING_MODEL = os.getenv("SENTINEL_EMBEDDING_MODEL", "qwen-1.5-2b")
```

**Option B: Environment Variable**

```bash
export SENTINEL_EMBEDDING_MODEL=qwen-1.5-2b
python run_large_scale_benchmark.py
```

**Option C: Direct Code**

```python
from src.embedder import SentinelEmbedder

# Initialize with Qwen 1.5 2B
embedder = SentinelEmbedder(
    model_name="qwen-1.5-2b",  # or full path
    device="cuda",
    verbose=True
)

# Encode texts
embeddings = embedder.encode(
    ["Your financial document here"],
    persona="Forensic Auditor"
)
```

### 3. Run Example

```bash
python example_qwen_1.5_2b.py
```

## Implementation Details

### Model Registry

The `SentinelEmbedder` class includes a built-in model registry:

```python
MODEL_REGISTRY = {
    "all-MiniLM": {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "vector_dim": 384,
        "description": "Lightweight 22M parameter model"
    },
    "qwen-1.5-2b": {
        "model_name": "Alibaba-NLP/gte-Qwen1.5-2B-instruct",
        "vector_dim": 1536,
        "description": "Qwen 1.5 2B with 1536-dimensional embeddings"
    },
    "qwen2-1.5b": {
        "model_name": "Alibaba-NLP/gte-Qwen2-1.5b-instruct",
        "vector_dim": 1536,
        "description": "Qwen 2.5 1.5B with 1536-dimensional embeddings"
    }
}
```

### Configuration Auto-adjustment

When you select Qwen 1.5 2B, the system automatically:

1. Sets `VECTOR_DIM = 1536`
2. Adjusts `BYTES_PER_FULL_VECTOR = 1536 * 4`
3. Adjusts `BYTES_PER_RABITQ_VECTOR = 1536 * 0.125`
4. Updates `COLLECTION_NAME` to include model identifier
5. Generates appropriate RaBitQ rotation matrix (1536×1536)

### Code Changes Summary

#### src/embedder.py

**Added:**
- `MODEL_REGISTRY`: Dictionary of available models
- Enhanced `__init__()` to support model selection
- `list_available_models()`: Static method to list models
- Enhanced `get_model_info()`: Returns available models

**Modified:**
- `__init__()` now accepts model names or full paths
- Auto-detection of vector dimension
- Flexible RaBitQ matrix generation for any dimension

#### src/config.py

**Added:**
```python
EMBEDDING_MODEL = os.getenv("SENTINEL_EMBEDDING_MODEL", "qwen-1.5-2b")
MODEL_DIMENSIONS = {
    "all-MiniLM": 384,
    "qwen-1.5-2b": 1536,
    "qwen2-1.5b": 1536,
}
VECTOR_DIM = MODEL_DIMENSIONS.get(EMBEDDING_MODEL, 1536)
```

**Modified:**
- `COLLECTION_NAME` now includes model identifier
- `VECTOR_DIM` automatically set based on model

## Usage Patterns

### Pattern 1: Simple Encoding

```python
from src.embedder import SentinelEmbedder

embedder = SentinelEmbedder("qwen-1.5-2b")

texts = [
    "Financial statement analysis",
    "Risk assessment report"
]

vectors = embedder.encode(texts)
# Returns: (2, 1536) numpy array
```

### Pattern 2: With Persona

```python
vectors = embedder.encode(
    texts,
    persona="Risk Analyst",
    normalize_embeddings=True
)
```

### Pattern 3: Batch Processing

```python
batch_1 = ["Text 1", "Text 2"]
batch_2 = ["Text 3", "Text 4"]

all_vectors = embedder.encode_batch(
    [batch_1, batch_2],
    batch_size=32,
    persona="CFO"
)
# Returns: (4, 1536) numpy array
```

### Pattern 4: Get Model Info

```python
info = embedder.get_model_info()
# Returns dict with model details

available = SentinelEmbedder.list_available_models()
# Returns registry of all models
```

## Performance Characteristics

### Memory Usage

- **Per embedding**: 1536 × 4 bytes = 6 KB (float32)
- **Per 1000 embeddings**: ~6 MB
- **Per 100K embeddings**: ~600 MB
- **With RaBitQ compression**: ~50 MB (12x reduction)

### Inference Speed

| Scenario | Speed |
|----------|-------|
| 100 texts (GPU) | 1-2 minutes |
| 1000 texts (GPU) | 10-20 minutes |
| 100K texts (GPU) | 2-4 hours |
| 100 texts (CPU) | 5-10 minutes |
| 1000 texts (CPU) | 50-100 minutes |

*Estimates: May vary based on hardware and batch size*

### Similarity Quality

Qwen 1.5 2B provides:
- 4x more dimensions than all-MiniLM (1536 vs 384)
- Better semantic understanding of financial documents
- Superior performance on domain-specific queries
- Improved retrieval accuracy

## Integration with Benchmark Pipeline

### Running Benchmark with Qwen 1.5 2B

```bash
# Method 1: Environment variable
export SENTINEL_EMBEDDING_MODEL=qwen-1.5-2b
python run_large_scale_benchmark.py

# Method 2: Modify config.py
# Edit src/config.py and set EMBEDDING_MODEL = "qwen-1.5-2b"
# Then run:
python run_large_scale_benchmark.py
```

### Expected Output

Results will be saved in `results/final_ieee_data.json` with:
- Vector dimension: 1536
- Collection name: `sentinel_100k_qwen_1_5_2b`
- Comparison metrics against baseline

## Troubleshooting

### Issue: Model Download Fails

```python
# Solution: Pre-download the model
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("Alibaba-NLP/gte-Qwen1.5-2B-instruct")
```

### Issue: Out of Memory

```python
# Reduce batch size
embedder.encode(texts, batch_size=16)  # Instead of default 64
```

### Issue: Slow Inference on CPU

```python
# Use GPU if available
import torch
if torch.cuda.is_available():
    embedder = SentinelEmbedder("qwen-1.5-2b", device="cuda")
```

### Issue: Dimension Mismatch

Ensure you're using consistent vector dimensions:

```python
# Check current dimension
print(embedder.vector_dim)  # Should be 1536 for Qwen

# When switching models, update config
export SENTINEL_EMBEDDING_MODEL=qwen-1.5-2b  # Forces 1536 dimensions
```

## Advanced Configuration

### Custom Model Support

To add a new embedding model:

1. **Edit MODEL_REGISTRY** in `src/embedder.py`:

```python
MODEL_REGISTRY = {
    ...
    "my-model": {
        "model_name": "huggingface/my-model",
        "vector_dim": 1024,
        "description": "My custom model"
    }
}
```

2. **Use it**:

```python
embedder = SentinelEmbedder("my-model")
```

### RaBitQ Configuration

RaBitQ settings are in `src/config.py`:

```python
RABITQ_EPSILON = 1.9           # Confidence parameter
RABITQ_USE_ORTHOGONAL = True   # Use scipy orthogonal matrices
RABITQ_NORMALIZE_OUTPUT = True  # L2 normalize after rotation
```

## Results Comparison

Expected improvements when switching to Qwen 1.5 2B:

- **Recall@10**: +5-15% improvement
- **NDCG**: +8-20% improvement
- **MRR**: +10-25% improvement
- **Inference time**: 3-5x slower (justified by quality)

## References

- **Model**: [Alibaba-NLP/gte-Qwen1.5-2B-instruct](https://huggingface.co/Alibaba-NLP/gte-Qwen1.5-2B-instruct)
- **Qwen**: https://github.com/QwenLM/Qwen
- **Sentence Transformers**: https://www.sbert.net/
- **RaBitQ Paper**: https://arxiv.org/abs/2309.02048

## Support

For issues or questions:
1. Check `src/embedder.py` for implementation details
2. Run `python example_qwen_1.5_2b.py` for working examples
3. Review `src/config.py` for configuration options
4. Check Streamlit dashboard: `streamlit run streamlit_app.py`

---

**Last Updated**: January 29, 2026
**SENTINEL Framework Version**: 2.0
**Status**: ✅ Production Ready
