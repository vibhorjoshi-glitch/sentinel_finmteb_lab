import json
import numpy as np
import time
from tqdm import tqdm
from src.dataset import load_financial_corpus
from src.embedder import FinancialEmbedder
from src.engine import SentinelEngine
from src.metrics import compute_fidelity_with_qrels, calculate_network_load
from src.config import RESULTS_PATH, GT_COLLECTION, BQ_COLLECTION
from qdrant_client import models

def run_experiment():
    print("===============================================================")
    print("   SENTINEL: FinMTEB Sovereign Retrieval Benchmark (IEEE)      ")
    print("===============================================================")

    # 1. Load Full Financial Manifold (With Qrels/Ground Truth)
    corpus, queries, qrels = load_financial_corpus(use_full_data=False)  # False for testing
    
    doc_ids = list(corpus.keys())
    doc_texts = list(corpus.values())
    
    query_ids = list(queries.keys())
    query_texts = list(queries.values())
    
    # 2. Vectorize
    embedder = FinancialEmbedder()
    
    print(f"\n[1/4] Vectorizing {len(doc_texts)} Financial Documents...")
    doc_vectors = embedder.encode(doc_texts, batch_size=64)
    
    print(f"[2/4] Vectorizing {len(query_texts)} Probe Queries...")
    query_vectors = embedder.encode(query_texts, batch_size=64)
    
    # 3. Ingest into Qdrant
    engine = SentinelEngine()
    engine.init_collections()
    
    print("\n[3/4] Ingesting into Qdrant (Float32 vs Binary)...")
    engine.ingest(doc_vectors, doc_texts, ids=doc_ids)
    
    # 4. Bandwidth Simulation
    cloud_bw, edge_bw = calculate_network_load(len(query_texts))
    print(f"\n[Network Physics] Cloud: {cloud_bw:.2f} Gbps | Sovereign: {edge_bw:.2f} Gbps")
    
    # 5. Run Fidelity Benchmark
    print("\n[4/4] Benchmarking Retrieval Fidelity (Recall@10)...")
    results = {}
    oversampling_factors = [1.0, 2.0, 4.0] 
    
    for factor in oversampling_factors:
        recalls = []
        hits = [] 
        start_time = time.time()
        
        for i, q_vec in enumerate(tqdm(query_vectors, desc=f"Oversample {factor}x")):
            current_qid = query_ids[i]
            
            # --- FIX: Use query_points() instead of search() ---
            bq_response = engine.client.query_points(
                collection_name=BQ_COLLECTION, 
                query=q_vec,
                limit=int(10 * factor),
                search_params=models.SearchParams(
                    quantization=models.QuantizationSearchParams(
                        ignore=False, 
                        rescore=True, 
                        oversampling=factor
                    )
                )
            )
            # Access .points to get the list of ScoredPoints
            retrieved_ids = [str(hit.id) for hit in bq_response.points][:10]
            
            # --- Compare against Ground Truth ---
            recall, is_hit = compute_fidelity_with_qrels(qrels, current_qid, retrieved_ids)
            
            recalls.append(recall)
            hits.append(1 if is_hit else 0)
            
        # Aggregation
        avg_recall = np.mean(recalls)
        mrr_score = np.mean(hits)
        elapsed = time.time() - start_time
        
        print(f"-> Factor {factor}x | Recall@10: {avg_recall:.4f} | MRR: {mrr_score:.4f} | Time: {elapsed:.1f}s")
        
        results[f"Oversample_{factor}"] = {
            "recall": avg_recall, 
            "mrr": mrr_score,
            "latency_avg_ms": (elapsed / len(queries)) * 1000,
            "bandwidth_reduction": cloud_bw / edge_bw
        }

    # 6. Save IEEE Report
    with open(f"{RESULTS_PATH}/ieee_finmteb_final_report.json", "w") as f:
        json.dump(results, f, indent=4)
        print(f"\n[Success] Data saved to {RESULTS_PATH}/ieee_finmteb_final_report.json")

if __name__ == "__main__":
    run_experiment()
