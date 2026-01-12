import numpy as np

def calculate_network_impact(n_queries, k=10, vec_dim=1024):
    """
    Calculates the Backhaul Traffic Reduction for IEEE TMLCN.
    
    Scenario:
    - Cloud Mode: Edge sends Query Vector (4KB) -> Cloud sends 10 Result Vectors (40KB).
    - Sentinel Mode: Edge does local Retrieval -> Sends only Final Text Verdict (0.5KB).
    """
    # 1. Physics Constants
    FLOAT32_BYTES = 4
    BINARY_BYTES = 0.125  # 1 bit
    METADATA_BYTES = 500  # Avg size of a JSON text verdict
    
    # 2. Cloud Architecture (Vector Offloading)
    # Traffic = Uplink (Query) + Downlink (Top-K Vectors + Metadata)
    cloud_uplink = n_queries * (vec_dim * FLOAT32_BYTES)
    cloud_downlink = n_queries * k * (vec_dim * FLOAT32_BYTES + METADATA_BYTES)
    total_cloud_bytes = cloud_uplink + cloud_downlink
    
    # 3. Sentinel Edge Architecture (Semantic Offloading)
    # Traffic = Uplink (Text Query) + Downlink (Text Answer) - No Vectors transmitted!
    # We assume purely text-based intent communication
    edge_uplink = n_queries * 100  # Approx 100 bytes for text query
    edge_downlink = n_queries * METADATA_BYTES  # Final text answer
    total_edge_bytes = edge_uplink + edge_downlink
    
    # 4. Conversion to Gbps (assuming 1-second burst for 10k users)
    cloud_gbps = (total_cloud_bytes * 8) / (1024**3)
    edge_gbps = (total_edge_bytes * 8) / (1024**3)
    
    reduction_factor = cloud_gbps / edge_gbps if edge_gbps > 0 else 0
    
    return {
        "cloud_gbps": cloud_gbps,
        "edge_gbps": edge_gbps,
        "reduction_factor": reduction_factor,
        "bandwidth_saved_percent": (1 - (edge_gbps / cloud_gbps)) * 100
    }


def compute_topological_integrity(qrels, query_id, retrieved_ids):
    """
    Measures 'Retrieval Integrity': Does the 1-bit Edge Topology matches the Cloud Topology?
    Uses 'Gold Standard' Qrels from FinMTEB.
    """
    ground_truth = set(qrels.get(query_id, []))
    
    if not ground_truth:
        return 0.0, False  # Skip invalid queries

    # Recall calculation
    retrieved_set = set(retrieved_ids)
    intersection = len(ground_truth.intersection(retrieved_set))
    recall = intersection / len(ground_truth)
    
    # Hit calculation (for MRR)
    is_hit = False
    if retrieved_ids and retrieved_ids[0] in ground_truth:
        is_hit = True
        
    return recall, is_hit
