"""
SENTINEL Dataset Manager
Smart subset loading from FiQA with guaranteed answerable queries
"""

import json
import logging
import os
import random
from typing import Dict, Tuple
from datasets import load_dataset

from src.config import N_SAMPLES

logger = logging.getLogger(__name__)


class SentinelDatasetManager:
    def __init__(self, cache_dir="data/cache", use_cache=True, verbose=False):
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.verbose = verbose
    
    def _cache_paths(self, target_docs):
        cache_key = f"finmteb_{target_docs}"
        return {
            "corpus": os.path.join(self.cache_dir, f"{cache_key}_corpus.json"),
            "queries": os.path.join(self.cache_dir, f"{cache_key}_queries.json"),
            "qrels": os.path.join(self.cache_dir, f"{cache_key}_qrels.json"),
        }
    
    def _load_cache(self, paths):
        with open(paths["corpus"], "r") as f:
            corpus = json.load(f)
        with open(paths["queries"], "r") as f:
            queries = json.load(f)
        with open(paths["qrels"], "r") as f:
            qrels = json.load(f)
        return corpus, queries, qrels
    
    def _save_cache(self, paths, corpus, queries, qrels):
        os.makedirs(self.cache_dir, exist_ok=True)
        with open(paths["corpus"], "w") as f:
            json.dump(corpus, f)
        with open(paths["queries"], "w") as f:
            json.dump(queries, f)
        with open(paths["qrels"], "w") as f:
            json.dump(qrels, f)
    
    def load_smart_subset(self, target_docs, loading_method="cached"):
        paths = self._cache_paths(target_docs)
        if (
            self.use_cache
            and loading_method == "cached"
            and all(os.path.exists(path) for path in paths.values())
        ):
            if self.verbose:
                print("   ✓ Loading dataset from cache")
            return self._load_cache(paths)
        
        if self.verbose:
            print("   -> Loading FiQA corpus, queries, and qrels from HuggingFace")
        
        corpus_ds = load_dataset("mteb/fiqa", "corpus", split="corpus")
        queries_ds = load_dataset("mteb/fiqa", "queries", split="queries")
        qrels_ds = load_dataset("mteb/fiqa", "default", split="test")
        
        qrels_map = {}
        doc_ids_needed = []
        seen_docs = set()
        
        for row in qrels_ds:
            qid = str(row["query-id"])
            did = str(row["corpus-id"])
            qrels_map.setdefault(qid, {})
            qrels_map[qid][did] = 1
            if did not in seen_docs:
                doc_ids_needed.append(did)
                seen_docs.add(did)
            if target_docs and len(doc_ids_needed) >= target_docs:
                break
        
        selected_doc_ids = set(doc_ids_needed)
        
        corpus = {}
        for row in corpus_ds:
            row_id = str(row["_id"])
            if row_id in selected_doc_ids:
                corpus[row_id] = {"title": row["title"], "text": row["text"]}
                if len(corpus) >= len(selected_doc_ids):
                    break
        
        queries = {}
        for row in queries_ds:
            row_id = str(row["_id"])
            if row_id in qrels_map:
                queries[row_id] = row["text"]
        
        if self.use_cache:
            self._save_cache(paths, corpus, queries, qrels_map)
        
        return corpus, queries, qrels_map
