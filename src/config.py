import os

N_SAMPLES = 100000        # The 100K target for your paper
VECTOR_DIM = 1536        # Qwen-2.5 GTE Dimension
RABITQ_EPSILON = 1.9     # 95% Confidence Bound for Rescoring

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "qdrant_storage")
COLLECTION_NAME = "sentinel_100k_manifold"

# --- FINANCIAL PERSONAS (Fin-E5 Methodology) ---
FINANCIAL_PERSONAS = {
    "Forensic Auditor": "Detect control failures, fraud indicators, anomalies",
    "Equity Analyst": "Assess revenue quality, margins, competitive positioning",
    "Risk Manager": "Identify market, credit, liquidity, operational risks",
    "Compliance Officer": "Verify regulatory adherence, governance, disclosures",
    "Tax Strategist": "Analyze tax efficiency, structuring, planning strategies"
}
