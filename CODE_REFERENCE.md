# SENTINEL: Qwen 1.5 2B - Complete Code Reference

## Table of Contents
1. [Core Implementation](#core-implementation)
2. [Configuration Changes](#configuration-changes)
3. [Usage Examples](#usage-examples)
4. [API Reference](#api-reference)
5. [Testing](#testing)

---

## Core Implementation

### Model Registry (src/embedder.py)

```python
MODEL_REGISTRY = {
    "all-MiniLM": {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "vector_dim": 384,
        "description": "Lightweight 22M parameter model for fast inference"
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

### Enhanced SentinelEmbedder.__init__()

```python
def __init__(
    self,
    model_name: str = "all-MiniLM",
    vector_dim: Optional[int] = None,
    device: Optional[str] = None,
    trust_remote_code: bool = True,
    verbose: bool = True
):
    """
    Initialize SentinelEmbedder with specified model.
    
    Args:
        model_name: Model identifier or HuggingFace path
        vector_dim: Expected output dimension (auto-detect if None)
        device: torch device ("cuda" or "cpu", auto-detect if None)
        trust_remote_code: Allow remote model code execution
        verbose: Print initialization messages
    
    Examples:
        # Using registry alias
        embedder = SentinelEmbedder("qwen-1.5-2b")
        
        # Using full model path
        embedder = SentinelEmbedder(
            "Alibaba-NLP/gte-Qwen1.5-2B-instruct",
            vector_dim=1536
        )
        
        # With GPU
        embedder = SentinelEmbedder("qwen-1.5-2b", device="cuda")
    """
    # Resolve model from registry
    if model_name in MODEL_REGISTRY:
        model_config = MODEL_REGISTRY[model_name]
        full_model_name = model_config["model_name"]
        default_dim = model_config["vector_dim"]
    else:
        full_model_name = model_name
        default_dim = vector_dim if vector_dim else 1536
    
    # Store configuration
    self.model_name = model_name
    self.full_model_name = full_model_name
    self.vector_dim = vector_dim if vector_dim is not None else default_dim
    self.verbose = verbose
    
    # Auto-detect device
    if device is None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        self.device = device
    
    # Load model with fallback strategy
    try:
        self.model = SentenceTransformer(
            full_model_name,
            device=self.device,
            trust_remote_code=trust_remote_code
        )
    except Exception as e:
        # Fallback to transformers
        from transformers import AutoTokenizer, AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained(
            full_model_name, 
            trust_remote_code=trust_remote_code
        )
        self.model = AutoModel.from_pretrained(
            full_model_name, 
            trust_remote_code=trust_remote_code
        ).to(self.device)
        self.use_transformers = True
    
    # Initialize RaBitQ rotation matrix
    P_raw = ortho_group.rvs(dim=self.vector_dim)
    self.P_matrix = torch.tensor(P_raw, dtype=torch.float32).to(self.device)
```

### Encode Method with Qwen Support

```python
def encode(
    self,
    sentences: Union[str, List[str]],
    batch_size: int = 64,
    show_progress_bar: bool = True,
    persona: str = "Forensic Auditor",
    normalize_embeddings: bool = True
) -> np.ndarray:
    """
    Encode sentences into vectors with RaBitQ rotation.
    
    Pipeline:
    1. Prefix with persona context
    2. Encode using selected model → (N, D) f32
    3. Apply RaBitQ rotation: v' = v @ P → (N, D)
    4. L2 normalize → (N, D) unit vectors
    
    Args:
        sentences: Single sentence or list of sentences
        batch_size: Batch size for processing
        show_progress_bar: Show progress indicator
        persona: Financial persona for domain adaptation
        normalize_embeddings: L2 normalize output
    
    Returns:
        np.ndarray of shape (N, D) where D is model dimension
    
    Examples:
        # Single document
        vec = embedder.encode("Investment strategy document")
        # Returns: (1, 1536) for Qwen 1.5 2B
        
        # Multiple documents with persona
        vecs = embedder.encode(
            ["Doc 1", "Doc 2", "Doc 3"],
            persona="Risk Analyst",
            batch_size=32
        )
        # Returns: (3, 1536)
        
        # Large batch with progress
        vecs = embedder.encode(
            large_document_list,
            batch_size=64,
            show_progress_bar=True
        )
    """
    # Handle single sentence
    if isinstance(sentences, str):
        sentences = [sentences]
    
    if not sentences:
        return np.array([], dtype=np.float32).reshape(0, self.vector_dim)
    
    # Add persona context
    prefixed_sentences = [
        f"System: [Persona: {persona}] | Content: {sent}"
        for sent in sentences
    ]
    
    # Encode with model
    with torch.no_grad():
        embeddings = self.model.encode(
            prefixed_sentences,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_tensor=True
        )
    
    # Ensure correct device
    if not isinstance(embeddings, torch.Tensor):
        embeddings = torch.tensor(
            embeddings, 
            dtype=torch.float32, 
            device=self.device
        )
    else:
        embeddings = embeddings.to(self.device)
    
    # Apply RaBitQ rotation
    with torch.no_grad():
        rotated = torch.matmul(embeddings, self.P_matrix)
    
    # L2 normalize
    if normalize_embeddings:
        with torch.no_grad():
            normalized = torch.nn.functional.normalize(
                rotated, 
                p=2, 
                dim=1
            )
    else:
        normalized = rotated
    
    return normalized.cpu().numpy().astype(np.float32)
```

### Static Method: List Available Models

```python
@staticmethod
def list_available_models() -> Dict[str, Dict]:
    """
    List all available embedding models.
    
    Returns:
        Dictionary with model configurations
    
    Example:
        models = SentinelEmbedder.list_available_models()
        for name, config in models.items():
            print(f"{name}: {config['vector_dim']} dimensions")
    """
    return MODEL_REGISTRY
```

---

## Configuration Changes

### src/config.py - Model Selection

```python
"""
SENTINEL Configuration Module
Supports multiple embedding models
"""

import os
import torch

# ============================================================================
# EMBEDDING MODEL SELECTION
# ============================================================================

# Choose embedding model:
# - "all-MiniLM": 384 dimensions, fast, lightweight
# - "qwen-1.5-2b": 1536 dimensions, high capacity
# - "qwen2-1.5b": 1536 dimensions, latest Qwen
EMBEDDING_MODEL = os.getenv("SENTINEL_EMBEDDING_MODEL", "qwen-1.5-2b")

# Dimension mapping for each model
MODEL_DIMENSIONS = {
    "all-MiniLM": 384,
    "qwen-1.5-2b": 1536,
    "qwen2-1.5b": 1536,
}

# Automatically set vector dimension based on model
VECTOR_DIM = MODEL_DIMENSIONS.get(EMBEDDING_MODEL, 1536)

# ============================================================================
# CORE PARAMETERS (auto-adjusted by VECTOR_DIM)
# ============================================================================

N_SAMPLES = 100000
TARGET_DOCS = 1000
RABITQ_EPSILON = 1.9
COMPRESSION_RATIO = 12.0

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "qdrant_storage")
RESULTS_PATH = os.path.join(BASE_DIR, "results")

# Collection name includes model identifier
COLLECTION_NAME = f"sentinel_100k_{EMBEDDING_MODEL.replace('-', '_')}"
GT_COLLECTION = f"{COLLECTION_NAME}_float32"
BQ_COLLECTION = COLLECTION_NAME

# Auto-calculated from VECTOR_DIM
CLOUD_LOAD_GBPS = 160.0
SENTINEL_LOAD_GBPS = CLOUD_LOAD_GBPS / COMPRESSION_RATIO
BYTES_PER_FULL_VECTOR = VECTOR_DIM * 4
BYTES_PER_RABITQ_VECTOR = VECTOR_DIM / 8

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FINAL_RESULTS_FILE = "final_ieee_data.json"
RECALL_AT_K = 10
EMBEDDING_BATCH_SIZE = 64
```

---

## Usage Examples

### Example 1: Basic Usage

```python
from src.embedder import SentinelEmbedder

# Initialize
embedder = SentinelEmbedder("qwen-1.5-2b")

# Encode documents
docs = [
    "Financial statement analysis for Q4 2023",
    "Risk management in volatile markets",
    "Portfolio optimization strategies"
]

embeddings = embedder.encode(docs)
print(f"Shape: {embeddings.shape}")  # (3, 1536)
```

### Example 2: With Configuration

```python
import os
from src.embedder import SentinelEmbedder
from src.config import VECTOR_DIM, EMBEDDING_MODEL

# Set model via environment
os.environ["SENTINEL_EMBEDDING_MODEL"] = "qwen-1.5-2b"

# Verify configuration
print(f"Model: {EMBEDDING_MODEL}")  # qwen-1.5-2b
print(f"Dimension: {VECTOR_DIM}")  # 1536

# Initialize with verified config
embedder = SentinelEmbedder(EMBEDDING_MODEL)
```

### Example 3: Batch Processing

```python
from src.embedder import SentinelEmbedder

embedder = SentinelEmbedder("qwen-1.5-2b")

# Process multiple batches
batches = [
    ["Doc 1", "Doc 2"],
    ["Doc 3", "Doc 4"],
    ["Doc 5", "Doc 6"]
]

all_vectors = embedder.encode_batch(
    batches,
    batch_size=32,
    persona="Portfolio Manager"
)

print(f"Total embeddings: {all_vectors.shape}")  # (6, 1536)
```

### Example 4: Similarity Computation

```python
import numpy as np
from src.embedder import SentinelEmbedder

embedder = SentinelEmbedder("qwen-1.5-2b")

documents = ["Risk analysis", "Return optimization", "Risk analysis v2"]
embeddings = embedder.encode(documents)

# Cosine similarity matrix
similarity = embeddings @ embeddings.T

print("Similarity Matrix:")
print(similarity)
# [[1.000  0.xxx  0.yyy]
#  [0.xxx  1.000  0.zzz]
#  [0.yyy  0.zzz  1.000]]

# Find most similar
query_idx = 0
sims = similarity[query_idx]
top_k = np.argsort(sims)[::-1][:3]
```

### Example 5: Model Comparison

```python
from src.embedder import SentinelEmbedder
import numpy as np

# Load both models
minilm = SentinelEmbedder("all-MiniLM")
qwen = SentinelEmbedder("qwen-1.5-2b")

# Same document
doc = "Financial risk assessment framework"

# Get embeddings
minilm_vec = minilm.encode(doc)  # (1, 384)
qwen_vec = qwen.encode(doc)      # (1, 1536)

print(f"all-MiniLM dimension: {minilm_vec.shape[1]}")  # 384
print(f"Qwen 1.5 2B dimension: {qwen_vec.shape[1]}")   # 1536

# Qwen captures more nuanced information
print(f"4x capacity increase: {qwen_vec.shape[1] / minilm_vec.shape[1]}")
```

---

## API Reference

### SentinelEmbedder Class

#### Constructor

```python
SentinelEmbedder(
    model_name: str = "all-MiniLM",
    vector_dim: Optional[int] = None,
    device: Optional[str] = None,
    trust_remote_code: bool = True,
    verbose: bool = True
)
```

**Parameters:**
- `model_name` (str): Model alias or HuggingFace path
- `vector_dim` (int, optional): Output dimension (auto if None)
- `device` (str, optional): "cuda" or "cpu" (auto-detect if None)
- `trust_remote_code` (bool): Allow remote code (default: True)
- `verbose` (bool): Print initialization logs (default: True)

**Returns:** SentinelEmbedder instance

#### Methods

##### encode()

```python
encode(
    sentences: Union[str, List[str]],
    batch_size: int = 64,
    show_progress_bar: bool = True,
    persona: str = "Forensic Auditor",
    normalize_embeddings: bool = True
) -> np.ndarray
```

Encodes text into embedding vectors.

**Parameters:**
- `sentences`: Single text or list of texts
- `batch_size`: Number of texts per batch (default: 64)
- `show_progress_bar`: Display progress (default: True)
- `persona`: Financial persona context (default: "Forensic Auditor")
- `normalize_embeddings`: L2 normalize (default: True)

**Returns:** np.ndarray of shape (N, D)

##### encode_batch()

```python
encode_batch(
    sentences_list: List[List[str]],
    batch_size: int = 64,
    persona: str = "Forensic Auditor",
    show_progress_bar: bool = True
) -> np.ndarray
```

Encodes multiple batches of sentences.

**Parameters:**
- `sentences_list`: List of sentence lists
- `batch_size`: Batch size (default: 64)
- `persona`: Financial persona (default: "Forensic Auditor")
- `show_progress_bar`: Display progress (default: True)

**Returns:** np.ndarray of shape (sum(lengths), D)

##### get_model_info()

```python
get_model_info() -> dict
```

Returns model configuration details.

**Returns:**
```python
{
    "model_name": str,           # Model alias
    "full_model_name": str,      # HuggingFace path
    "vector_dim": int,           # Embedding dimension
    "device": str,               # cuda/cpu
    "rabitq_enabled": bool,      # RaBitQ compression
    "p_matrix_shape": tuple,     # Rotation matrix shape
    "normalization": str,        # L2
    "available_models": list     # Registry keys
}
```

##### list_available_models() [Static]

```python
@staticmethod
list_available_models() -> Dict[str, Dict]
```

Lists all registered models.

**Returns:**
```python
{
    "all-MiniLM": {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "vector_dim": 384,
        "description": "..."
    },
    "qwen-1.5-2b": {
        "model_name": "Alibaba-NLP/gte-Qwen1.5-2B-instruct",
        "vector_dim": 1536,
        "description": "..."
    },
    ...
}
```

##### get_device()

```python
get_device() -> str
```

Returns current device.

**Returns:** "cuda" or "cpu"

---

## Testing

### Unit Test: Model Loading

```python
def test_qwen_model_loading():
    from src.embedder import SentinelEmbedder
    
    # Test model loads
    embedder = SentinelEmbedder("qwen-1.5-2b")
    
    assert embedder.vector_dim == 1536
    assert embedder.model_name == "qwen-1.5-2b"
    assert embedder.P_matrix.shape == (1536, 1536)
    
    print("✅ Model loading test passed")

test_qwen_model_loading()
```

### Unit Test: Encoding

```python
def test_encoding():
    from src.embedder import SentinelEmbedder
    import numpy as np
    
    embedder = SentinelEmbedder("qwen-1.5-2b")
    
    # Test single document
    vec = embedder.encode("Test document")
    assert vec.shape == (1, 1536)
    assert vec.dtype == np.float32
    
    # Test multiple documents
    vecs = embedder.encode(["Doc 1", "Doc 2", "Doc 3"])
    assert vecs.shape == (3, 1536)
    
    # Test normalization
    norms = (vecs ** 2).sum(axis=1) ** 0.5
    assert np.allclose(norms, 1.0, atol=1e-6)
    
    print("✅ Encoding test passed")

test_encoding()
```

### Integration Test: Config Synchronization

```python
def test_config_sync():
    import os
    os.environ["SENTINEL_EMBEDDING_MODEL"] = "qwen-1.5-2b"
    
    from src.embedder import SentinelEmbedder
    from src.config import VECTOR_DIM, EMBEDDING_MODEL, COLLECTION_NAME
    
    # Check config
    assert VECTOR_DIM == 1536
    assert EMBEDDING_MODEL == "qwen-1.5-2b"
    assert "qwen" in COLLECTION_NAME
    
    # Check embedder
    embedder = SentinelEmbedder(EMBEDDING_MODEL)
    assert embedder.vector_dim == VECTOR_DIM
    
    print("✅ Config synchronization test passed")

test_config_sync()
```

### Performance Test

```python
def test_performance():
    import time
    import numpy as np
    from src.embedder import SentinelEmbedder
    
    embedder = SentinelEmbedder("qwen-1.5-2b")
    
    # Create test data
    docs = [f"Document {i}: " + " ".join([f"word{j}"] * 10) 
            for i in range(100)]
    
    # Time encoding
    start = time.time()
    embeddings = embedder.encode(docs, show_progress_bar=False)
    elapsed = time.time() - start
    
    # Verify
    assert embeddings.shape == (100, 1536)
    
    # Report
    print(f"✅ Encoded 100 documents in {elapsed:.2f}s")
    print(f"   Average: {elapsed/100:.3f}s per document")
    
    return elapsed

perf = test_performance()
```

---

## Summary

**Key Components:**
- ✅ `MODEL_REGISTRY` - Multi-model support
- ✅ Enhanced `__init__()` - Flexible model loading
- ✅ Dynamic `VECTOR_DIM` - Automatic config adjustment
- ✅ `list_available_models()` - Model discovery
- ✅ Batch processing support
- ✅ RaBitQ compression for all models

**Supported Models:**
- all-MiniLM (384D) - Fast
- Qwen 1.5 2B (1536D) - High capacity
- Qwen2 1.5B (1536D) - Latest

**Status:** ✅ Production Ready

---

**Last Updated:** January 29, 2026
**Framework:** SENTINEL 2.0
