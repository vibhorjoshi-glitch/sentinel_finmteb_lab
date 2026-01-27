"""
SENTINEL Configuration Module
Centralized configuration for IEEE TMLCN Final Benchmark
"""

import os
import torch

# ============================================================================
# CORE SYSTEM PARAMETERS
# ============================================================================

# Target scale for IEEE paper
N_SAMPLES = 50000  # 50K documents benchmark (large-scale)
TARGET_DOCS = 50000  # Full-scale benchmark with ground truth
N_QUERIES = 50000  # 50K queries for comprehensive evaluation

# Embedding model parameters
VECTOR_DIM = 1024  # BGE-large output dimension
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
EMBEDDING_BATCH_SIZE = 64

# RaBitQ compression parameters
RABITQ_EPSILON = 1.9  # 95% confidence bound (Johnson-Lindenstrauss)
COMPRESSION_RATIO = 32.0  # 1536 dims × 4 bytes × (1/32) = 19.2 MB per 100K

# ============================================================================
# STORAGE CONFIGURATION
# ============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "qdrant_storage")
RESULTS_PATH = os.path.join(BASE_DIR, "results")

# Ensure directories exist
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)

# Qdrant collection configuration
COLLECTION_NAME = "sentinel_100k_manifold"
QDRANT_TIMEOUT = 30
QDRANT_PREFER_GRPC = False

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================

# FiQA corpus details
DATASET_NAME = "mteb/fiqa"
CORPUS_SPLIT = "corpus"
QUERIES_SPLIT = "queries"
QRELS_SPLIT = "qrels"

# Field names in FiQA dataset
CORPUS_TITLE_FIELD = "title"
CORPUS_TEXT_FIELD = "text"
QUERY_TEXT_FIELD = "text"

# ============================================================================
# BENCHMARK PARAMETERS
# ============================================================================

# Retrieval metrics
RECALL_AT_K = 10
RETRIEVAL_TOP_K = 20  # Retrieve more than needed for confidence-driven rescoring
CONFIDENCE_THRESHOLD = 0.5  # RaBitQ confidence threshold for rescoring

# Oversampling for fidelity analysis
OVERSAMPLING_FACTORS = [1, 2, 3, 4]  # For fidelity vs compression trade-off

# ============================================================================
# NETWORK LOAD CALCULATION
# ============================================================================

# Per-query network analysis
BYTES_PER_FULL_VECTOR = 1536 * 4  # f32: 6144 bytes per vector
BYTES_PER_RABITQ_VECTOR = 1536 * 0.125  # 1-bit: 192 bytes per vector
BYTES_PER_QUERY_RESULT = 200  # Verdict/answer only

# Baseline cloud-centric load (at 10K concurrent nodes)
CLOUD_LOAD_GBPS = 160.0
SENTINEL_LOAD_GBPS = 5.0

# ============================================================================
# FINANCIAL PERSONAS (Domain Adaptation)
# ============================================================================

FINANCIAL_PERSONAS = {
    "Forensic Auditor": "Specializes in detecting financial irregularities, fraud patterns, and regulatory violations.",
    "Risk Analyst": "Focuses on identifying market risks, credit risks, and operational vulnerabilities.",
    "Compliance Officer": "Ensures adherence to regulations, documentation standards, and audit trails.",
    "Portfolio Manager": "Analyzes investment opportunities, return metrics, and risk-adjusted performance.",
    "CFO": "Evaluates financial health, cash flows, capital allocation, and strategic decisions.",
}

DEFAULT_PERSONA = "Forensic Auditor"

# ============================================================================
# DEVICE & COMPUTE
# ============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float32
NUM_WORKERS = 4

# ============================================================================
# LOGGING & OUTPUT
# ============================================================================

LOG_LEVEL = "INFO"
VERBOSE = True
SAVE_INTERMEDIATE_RESULTS = True

# Output file names
FINAL_RESULTS_FILE = "final_ieee_data.json"
RESULTS_TABLE_FILE = "SENTINEL_RESULTS_TABLE.md"

# ============================================================================
# ADVANCED CONFIGURATION (Do not modify unless experienced)
# ============================================================================

# Qdrant on-disk storage (critical for 100K scale)
QDRANT_ON_DISK = True
QDRANT_BINARY_QUANTIZATION = True
QDRANT_ALWAYS_RAM = True

# Batch processing
MAX_BATCH_SIZE_ENCODE = 512
MAX_BATCH_SIZE_RETRIEVAL = 32

# RaBitQ implementation
RABITQ_USE_ORTHOGONAL = True  # Use scipy.stats.ortho_group for maximum robustness
RABITQ_NORMALIZE_OUTPUT = True  # L2 normalize after rotation

# ============================================================================
# ASSERTION CHECKS (Validate configuration)
# ============================================================================

assert VECTOR_DIM == 1024, "VECTOR_DIM must be 1024 (BGE-large output)"
assert COMPRESSION_RATIO == 32.0, "COMPRESSION_RATIO must be 32.0"
assert RABITQ_EPSILON > 0, "RABITQ_EPSILON must be positive"
assert RECALL_AT_K > 0, "RECALL_AT_K must be positive"
assert DEVICE in ["cuda", "cpu"], "DEVICE must be 'cuda' or 'cpu'"

# ============================================================================
# PRINT CONFIGURATION ON IMPORT (if VERBOSE)
# ============================================================================

if VERBOSE:
    print("=" * 70)
    print("SENTINEL CONFIGURATION LOADED")
    print("=" * 70)
    print(f"Target Scale (Paper): {N_SAMPLES:,} documents")
    print(f"Benchmark Scale: {TARGET_DOCS:,} documents with ground truth")
    print(f"Embedding Model: {EMBEDDING_MODEL}")
    print(f"Vector Dimension: {VECTOR_DIM}")
    print(f"Compression Ratio: {COMPRESSION_RATIO}x")
    print(f"Device: {DEVICE}")
    print(f"Data Path: {DATA_PATH}")
    print(f"Results Path: {RESULTS_PATH}")
    print("=" * 70)
