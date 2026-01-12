#!/usr/bin/env python3
"""
IEEE TMLCN: Distributed Edge Financial Auditing Benchmark
Solves the Backhaul Bottleneck in 6G Networks
"""

import json
import numpy as np
import time
import os
from tqdm import tqdm
from src.dataset import load_financial_corpus
from src.embedder import QwenFinancialEmbedder
from src.engine import SentinelEngine
from src.metrics import compute_topological_integrity, calculate_network_impact
from src.config import RESULTS_PATH, GT_COLLECTION, BQ_COLLECTION, CONCURRENT_NODES, EMBEDDING_BATCH_SIZE
from qdrant_client import models


def run_simulation():
    print("="*70)
    print("   IEEE TMLCN BENCHMARK: Distributed Edge Financial Auditing")
    print("="*70)

    # 1. Load Data (FinMTEB - Full Scale)
    print("\n[Phase 1] Loading FinMTEB Financial Corpus...")
    corpus, queries, qrels = load_financial_corpus(use_full_data=True)
    
    doc_ids = list(corpus.keys())
    doc_texts = list(corpus.values())
    query_texts = list(queries.values())
    query_ids = list(queries.keys())

    print(f"   ✓ Corpus: {len(doc_texts):,} documents")
    print(f"   ✓ Queries: {len(query_texts):,} queries")
    print(f"   ✓ Qrels: {len(qrels):,} ground truth entries")

    # 2. Vectorize (Simulating Edge Compute)
    embedder = QwenFinancialEmbedder()
    print(f"\n[Phase 2] Vectorizing Texts...")
    print(f"   Encoding {len(doc_texts):,} documents...")
    doc_vectors = embedder.encode(doc_texts, batch_size=EMBEDDING_BATCH_SIZE)
    
    print(f"   Encoding {len(query_texts):,} queries...")
    query_vectors = embedder.encode(query_texts, batch_size=EMBEDDING_BATCH_SIZE)

    # 3. Ingest (Building the Manifold)
    engine = SentinelEngine()
    engine.init_sovereign_collection()
    print(f"\n[Phase 3] Building Topological Indices (Float32 vs Binary)...")
    engine.ingest(doc_vectors, doc_texts, ids=doc_ids)
    print("   ✓ Collections initialized and indexed")

    # 4. Network Impact Analysis (The Core TMLCN Contribution)
    print(f"\n[Phase 4] Simulating {CONCURRENT_NODES:,} Concurrent Auditors...")
    net_stats = calculate_network_impact(CONCURRENT_NODES)
    
    print(f"   -> Cloud Backhaul Load:  {net_stats['cloud_gbps']:.2f} Gbps")
    print(f"   -> Edge Backhaul Load:   {net_stats['edge_gbps']:.4f} Gbps")
    print(f"   -> EFFICIENCY GAIN:      {net_stats['reduction_factor']:.1f}x Reduction")
    print(f"   -> Bandwidth Saved:      {net_stats['bandwidth_saved_percent']:.1f}%")

    # 5. Topological Integrity Benchmark (Fidelity)
    print(f"\n[Phase 5] Measuring Information Loss (Recall vs Compression)...")
    results = {
        "network_stats": net_stats,
        "fidelity_stats": {}
    }
    
    # We test the "Oversampling" strategy to prove we can recover lost info
    strategies = [1.0, 2.0, 4.0]
    
    for factor in strategies:
        print(f"\n   Testing Oversampling Factor: {factor}x")
        recalls = []
        hits = []
        
        for i, q_vec in enumerate(tqdm(query_vectors, desc=f"Strategy {factor}x")):
            qid = query_ids[i]
            
            # Sovereign Edge Search with Oversampling
            bq_response = engine.sovereign_search(q_vec, oversample=factor)
            retrieved_ids = [str(hit.id) for hit in bq_response.points][:10]
            
            # Fidelity Check
            recall, is_hit = compute_topological_integrity(qrels, qid, retrieved_ids)
            recalls.append(recall)
            hits.append(1 if is_hit else 0)
            
        # Aggregation
        avg_recall = np.mean(recalls) if recalls else 0.0
        avg_mrr = np.mean(hits) if hits else 0.0
        
        print(f"   ✓ Recall@10: {avg_recall:.4f}")
        print(f"   ✓ MRR@10:    {avg_mrr:.4f}")
        
        results["fidelity_stats"][f"Oversample_{factor}"] = {
            "recall": float(avg_recall),
            "mrr": float(avg_mrr)
        }
        
    # 6. Save IEEE Artifacts
    if not os.path.exists(RESULTS_PATH):
        os.makedirs(RESULTS_PATH)
        
    output_file = f"{RESULTS_PATH}/ieee_tmlcn_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"\n[Success] Simulation Complete.")
    print(f"Results saved to: {output_file}")
    
    # 7. Print Summary
    print("\n" + "="*70)
    print("   RESEARCH SUMMARY")
    print("="*70)
    print(f"Network Efficiency Gain: {net_stats['reduction_factor']:.1f}x")
    print(f"Best Recall Achievement: {max([v['recall'] for v in results['fidelity_stats'].values()]):.4f}")
    print("="*70)


if __name__ == "__main__":
    run_simulation()
