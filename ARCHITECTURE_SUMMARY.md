# SENTINEL Architecture Overview

## Current State

### Source Code Structure (`src/`)

```
src/
├── __init__.py                    # Package init
├── config.py                      # Configuration constants
├── embedder.py                    # SentinelEmbedder class
├── engine.py                      # SentinelEngine class
├── dataset.py                     # Dataset loading (legacy)
├── metrics.py                     # Metrics (legacy)
├── agents.py                      # Agent system (legacy)
├── network.py                     # Network simulation (legacy)
└── sentinel_sovereign_lab.py       # Phase 2&3 implementation (legacy)
```

### Active Components

#### 1. **src/config.py** (Actively Used)
```python
N_SAMPLES = 100000                 # Target scale for paper
VECTOR_DIM = 1536                  # Qwen-2.5-1.5B-GTE output dimension
RABITQ_EPSILON = 1.9               # 95% confidence bound
BASE_DIR = ...
DATA_PATH = .../data/qdrant_storage
COLLECTION_NAME = "sentinel_100k_manifold"
FINANCIAL_PERSONAS = {...}         # 5-persona dictionary
```

**Used by**: All scripts load from this

---

#### 2. **src/embedder.py** (Actively Used)
```python
from sentence_transformers import SentenceTransformer
from scipy.stats import ortho_group
import torch

class SentinelEmbedder:
    def __init__(self):
        # Load: Qwen-2.5-1.5B-GTE on GPU/CPU
        self.model = SentenceTransformer("Alibaba-NLP/gte-Qwen2-1.5b-instruct", ...)
        self.model._first_module().auto_model.config.use_cache = False  # FIX
        
        # RaBitQ: 1536×1536 orthogonal matrix
        P_raw = ortho_group.rvs(dim=1536)
        self.P_matrix = torch.tensor(P_raw, dtype=torch.float32)
    
    def encode(self, texts, persona="Forensic Auditor"):
        # Returns: (N, 1536) normalized numpy array
```

**Used by**: 
- `run_ieee_final.py` → embedder.encode(docs)
- `run_sentinel_100k.py` → embedder.encode(docs)

---

#### 3. **src/engine.py** (Actively Used)
```python
from qdrant_client import QdrantClient, models

class SentinelEngine:
    def __init__(self):
        self.client = QdrantClient(path=DATA_PATH)
    
    def init_collection(self):
        # Creates Qdrant collection with:
        # - vectors_config: on_disk=True (SSD storage)
        # - quantization_config: BinaryQuantization (32x compression)
    
    def close(self):
        # Explicit close to prevent __del__ errors
```

**Used by**:
- `run_ieee_final.py` → engine.init_collection(), engine.client.upsert()
- `run_sentinel_100k.py` → engine.init_collection(), engine.client.upsert()

---

### What Gets Imported in run_ieee_final.py

```python
from qdrant_client import models  # Line 8
```

This imports the **models** module, which includes:
- `models.VectorParams` - Configuration for vectors
- `models.Distance` - Distance metrics (COSINE)
- `models.BinaryQuantization` - Binary quantization config
- `models.BinaryQuantizationConfig` - Specific 1-bit settings
- `models.Batch` - Batch upsert format

**Usage in engine.py**:
```python
# Line 12-19 in engine.py
models.VectorParams(
    size=VECTOR_DIM,
    distance=models.Distance.COSINE,
    on_disk=True  # ← THE KEY OPTIMIZATION
)

models.BinaryQuantization(
    binary=models.BinaryQuantizationConfig(always_ram=True)
)
```

**Usage in run_ieee_final.py**:
```python
# Line 125
models.Batch(ids=doc_ids, vectors=doc_vectors.tolist())
```

---

## Data Flow in run_ieee_final.py

```
1. Load Data (get_smart_subset)
   ↓
2. Initialize System
   ├── SentinelEmbedder()           ← Uses config.VECTOR_DIM
   ├── SentinelEngine()             ← Uses config.DATA_PATH, COLLECTION_NAME
   └── engine.init_collection()     ← Uses qdrant_client.models for config
   ↓
3. Vectorize Documents
   ├── embedder.encode(docs)        ← Returns (N, 1536) numpy
   └── engine.client.upsert()       ← Uses models.Batch
   ↓
4. Vectorize & Search Queries
   ├── embedder.encode(queries)     ← Returns (Q, 1536) numpy
   └── engine.confidence_driven_search()  ← Retrieves from Qdrant
   ↓
5. Calculate Metrics
   └── Save to results/final_ieee_data.json
   ↓
6. Cleanup
   └── engine.close()               ← Graceful shutdown
```

---

## Key Design Decisions

### 1. **on_disk=True** (Line 15 in engine.py)
- Raw float32 vectors stored on SSD
- Only 1-bit quantized vectors in RAM
- Result: 18.3 MB RAM for 100K docs (vs 585.9 MB)

### 2. **BinaryQuantization with always_ram=True** (Line 18)
- 1-bit index always available in RAM for speed
- Raw f32 vectors accessed from disk for rescoring
- Hybrid approach: fast retrieval + accurate rescoring

### 3. **RaBitQ Rotation** (in embedder.py)
- Random orthogonal matrix P: 1536×1536
- Transformation: v' = v @ P
- Guarantees: topological preservation despite 32x compression

### 4. **Persona Augmentation** (in embedder.py)
- Prefix: "System: [Persona: {persona}] | Content: {text}"
- Encodes financial perspective into vector
- Improves retrieval for domain-specific queries

---

## What's NOT Being Used (Legacy Code)

These exist but aren't needed for IEEE final benchmark:
- `agents.py` - SACAIR pipeline (not in final benchmark)
- `network.py` - Network simulation (already calculated in run_ieee_final.py)
- `metrics.py` - Custom metrics (using standard Recall@10)
- `dataset.py` - Legacy corpus loading
- `sentinel_sovereign_lab.py` - Earlier Phase 2&3 prototype

---

## Summary Table

| Component | File | Status | In Use |
|-----------|------|--------|--------|
| Configuration | `config.py` | ✅ Active | Yes - All scripts |
| Embedder | `embedder.py` | ✅ Active | Yes - Vectorization |
| Engine | `engine.py` | ✅ Active | Yes - Qdrant storage |
| Agents | `agents.py` | ⚪ Legacy | No |
| Network | `network.py` | ⚪ Legacy | No |
| Metrics | `metrics.py` | ⚪ Legacy | No |
| Dataset | `dataset.py` | ⚪ Legacy | No |

---

## Running the Final Benchmark

```bash
python run_ieee_final.py
```

This script:
1. Loads config from `src/config.py`
2. Uses `SentinelEmbedder` from `src/embedder.py`
3. Uses `SentinelEngine` from `src/engine.py`
4. Imports `models` from `qdrant_client` for batch operations
5. Outputs metrics to `results/final_ieee_data.json`

That's it! Clean, minimal, ready for production. 🚀
