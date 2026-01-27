"""
SENTINEL Dataset Manager
Smart subset loading from FiQA with guaranteed answerable queries
"""

import logging
import random
from typing import Dict, Tuple
from datasets import load_dataset

from src.config import TARGET_DOCS

logger = logging.getLogger(__name__)


class SentinelDatasetManager:
    """Dataset manager for FiQA with smart subset selection."""
    
    def __init__(
        self,
        dataset_name: str = "mteb/fiqa",
        cache_dir: str = "data/cache",
        use_cache: bool = True,
        verbose: bool = True
    ):
        """Initialize dataset manager."""
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.verbose = verbose
    
    def load_smart_subset(
        self,
        target_docs: int = TARGET_DOCS,
        loading_method: str = "cached"
    ) -> Tuple[Dict, Dict, Dict]:
        """
        Load smart subset of FiQA data.
        
        Guarantees:
        - All selected queries have relevant documents in corpus
        - Corpus includes all gold documents + random distractors
        - Enables Recall > 0 evaluation
        
        Args:
            target_docs: Target number of documents (N_SAMPLES)
            loading_method: Loading method (for compatibility)
        
        Returns:
            (corpus_dict, queries_dict, qrels_dict)
        """
        if self.verbose:
            logger.info(f"--- Loading FinMTEB (FiQA) Smart-Subset (target: {target_docs} docs) ---")
        
        # Step 1: Load raw datasets
        if self.verbose:
            logger.info("Step 1: Loading corpus, queries, and qrels from HuggingFace...")
        
        corpus_ds = load_dataset(self.dataset_name, "corpus", split="corpus")
        queries_ds = load_dataset(self.dataset_name, "queries", split="queries")
        qrels_ds = load_dataset(self.dataset_name, "default", split="test")
        
        if self.verbose:
            logger.info(f"  Loaded {len(corpus_ds)} docs, {len(queries_ds)} queries, {len(qrels_ds)} qrels")
        
        # Step 2: Build qrels map {query_id: [doc_id, doc_id, ...]}
        if self.verbose:
            logger.info("Step 2: Building qrels map...")
        
        qrels_dict = {}
        for row in qrels_ds:
            qid = str(row["query-id"])
            docid = str(row["corpus-id"])
            
            if qid not in qrels_dict:
                qrels_dict[qid] = []
            qrels_dict[qid].append(docid)
        
        if self.verbose:
            logger.info(f"  Built qrels map: {len(qrels_dict)} queries with answers")
        
        # Step 3: Smart subsetting - select answerable queries
        if self.verbose:
            logger.info("Step 3: Smart subsetting...")
        
        # Find queries with answers
        valid_qids = [q for q in list(qrels_dict.keys()) if len(qrels_dict[q]) > 0]
        
        # Take up to 50K queries (or all available if fewer)
        num_queries = min(50000, len(valid_qids))
        selected_query_ids = valid_qids[:num_queries]
        
        if self.verbose:
            logger.info(f"  Selected {len(selected_query_ids)} answerable queries")
        
        # Collect all gold documents (answers to selected queries)
        must_have_doc_ids = set()
        for qid in selected_query_ids:
            for doc_id in qrels_dict[qid]:
                must_have_doc_ids.add(doc_id)
        
        if self.verbose:
            logger.info(f"  Identified {len(must_have_doc_ids)} gold documents")
        
        # Fill remaining slots with random distractors
        remaining_slots = target_docs - len(must_have_doc_ids)
        if remaining_slots > 0:
            # Get all doc IDs
            all_doc_ids = [str(row["_id"]) for row in corpus_ds]
            
            # Pool = all docs minus gold docs
            pool = list(set(all_doc_ids) - must_have_doc_ids)
            
            # Sample distractors
            num_distractors = min(len(pool), remaining_slots)
            distractors = random.sample(pool, num_distractors)
            selected_doc_ids = must_have_doc_ids.union(set(distractors))
            
            if self.verbose:
                logger.info(f"  Added {num_distractors} random distractor documents")
        else:
            selected_doc_ids = must_have_doc_ids
        
        if self.verbose:
            logger.info(f"  Final subset: {len(selected_doc_ids)} documents")
        
        # Step 4: Filter datasets
        if self.verbose:
            logger.info("Step 4: Filtering datasets...")
        
        # Filter corpus
        corpus_dict = {}
        for row in corpus_ds:
            row_id = str(row["_id"])
            if row_id in selected_doc_ids:
                corpus_dict[row_id] = {
                    "title": row.get("title", ""),
                    "text": row.get("text", "")
                }
        
        if self.verbose:
            logger.info(f"  Filtered corpus: {len(corpus_dict)} documents")
        
        # Filter queries
        queries_dict = {}
        for row in queries_ds:
            row_id = str(row["_id"])
            if row_id in selected_query_ids:
                queries_dict[row_id] = row["text"]
        
        if self.verbose:
            logger.info(f"  Filtered queries: {len(queries_dict)} queries")
        
        # Filter qrels to only selected docs
        filtered_qrels = {}
        for qid in selected_query_ids:
            if qid in qrels_dict:
                # Keep only docs that are in filtered corpus
                filtered_rels = [doc_id for doc_id in qrels_dict[qid] if doc_id in corpus_dict]
                if filtered_rels:
                    filtered_qrels[qid] = {doc_id: 1.0 for doc_id in filtered_rels}
        
        if self.verbose:
            logger.info(f"  Filtered qrels: {len(filtered_qrels)} queries with ground truth")
            avg_rels = sum(len(v) for v in filtered_qrels.values()) / len(filtered_qrels) if filtered_qrels else 0
            logger.info(f"\n✅ SUBSET LOADED:\n  Docs: {len(corpus_dict)}\n  Queries: {len(queries_dict)}\n  Avg relevant docs/query: {avg_rels:.1f}\n")
        
        return corpus_dict, queries_dict, filtered_qrels
