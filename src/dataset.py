from datasets import load_dataset
import random
from .config import N_SAMPLES

def load_financial_corpus(use_full_data=False):
    """
    Loads FinMTEB (FiQA) but ensures 'Answerable' queries.
    It guarantees that the relevant documents for the chosen queries 
    are actually included in the N_SAMPLES corpus.
    """
    print(f"--- Loading FinMTEB (FiQA) Smart-Subset (Full Data: {use_full_data}) ---")
    
    # 1. Load Everything (Metadata only, fast)
    corpus_ds = load_dataset("mteb/fiqa", "corpus", split="corpus")
    queries_ds = load_dataset("mteb/fiqa", "queries", split="queries")
    qrels_ds = load_dataset("mteb/fiqa", "default", split="test")

    # 2. Build Qrels Map {query_id: [doc_id, doc_id]}
    qrels_dict = {}
    for row in qrels_ds:
        qid = str(row["query-id"])
        docid = str(row["corpus-id"])
        if qid not in qrels_dict:
            qrels_dict[qid] = []
        qrels_dict[qid].append(docid)

    # 3. Smart Subsetting
    if use_full_data:
        # Use everything
        selected_doc_ids = set(str(row["_id"]) for row in corpus_ds)
        selected_query_ids = list(qrels_dict.keys())
    else:
        # --- THE FIX ---
        # 1. Pick 100 queries that actually have answers
        valid_qids = [q for q in list(qrels_dict.keys()) if len(qrels_dict[q]) > 0]
        selected_query_ids = valid_qids[:100] 
        
        # 2. Collect all document IDs that are answers to these 100 queries
        # (This guarantees Recall > 0 is possible)
        must_have_doc_ids = set()
        for qid in selected_query_ids:
            for doc_id in qrels_dict[qid]:
                must_have_doc_ids.add(doc_id)
        
        print(f"   -> Identified {len(must_have_doc_ids)} 'Gold' documents required for these queries.")

        # 3. Fill the rest of N_SAMPLES with random documents (Distractors)
        # to simulate the "Haystack"
        remaining_slots = N_SAMPLES - len(must_have_doc_ids)
        if remaining_slots > 0:
            # Get all available doc IDs
            all_doc_ids = [str(row["_id"]) for row in corpus_ds]
            # Remove ones we already picked
            pool = list(set(all_doc_ids) - must_have_doc_ids)
            # Pick random distractors
            distractors = random.sample(pool, min(len(pool), remaining_slots))
            selected_doc_ids = must_have_doc_ids.union(set(distractors))
        else:
            selected_doc_ids = must_have_doc_ids

    # 4. Filter the Datasets based on our Smart Selection
    # (This takes a moment but ensures data consistency)
    
    print("   -> Filtering Corpus...")
    corpus_dict = {}
    # We iterate once to find our selected IDs
    # Note: For speed on large data, we use a set check
    for row in corpus_ds:
        row_id = str(row["_id"])
        if row_id in selected_doc_ids:
            corpus_dict[row_id] = f"{row['title']} {row['text']}"
            if len(corpus_dict) >= len(selected_doc_ids):
                break # Stop once we have all we need

    print("   -> Filtering Queries...")
    queries_dict = {}
    for row in queries_ds:
        row_id = str(row["_id"])
        if row_id in selected_query_ids:
            queries_dict[row_id] = row["text"]

    print(f"Loaded {len(corpus_dict)} docs, {len(queries_dict)} queries.")
    return corpus_dict, queries_dict, qrels_dict
