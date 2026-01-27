"""
SENTINEL Metrics Module
Comprehensive evaluation with 10+ metrics for ranking quality assessment
"""

import logging
import numpy as np
from typing import Dict, Set, List, Tuple
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


# ============================================================================
# INDIVIDUAL METRIC CALCULATORS
# ============================================================================

class RecallCalculator:
    """Calculate Recall@K metric."""
    
    @staticmethod
    def recall_at_k(relevant: Set, retrieved: Set, k: int = 10) -> float:
        """
        Recall@K = |relevant ∩ retrieved| / |relevant|
        
        Args:
            relevant: Set of relevant document IDs
            retrieved: Set of retrieved document IDs (top-k)
            k: Cutoff rank
        
        Returns:
            Recall value in [0, 1]
        """
        if not relevant:
            return 0.0
        
        hits = len(relevant & retrieved)
        return hits / len(relevant)


class PrecisionCalculator:
    """Calculate Precision@K metric."""
    
    @staticmethod
    def precision_at_k(relevant: Set, retrieved: Set, k: int = 10) -> float:
        """
        Precision@K = |relevant ∩ retrieved| / |retrieved|
        
        Args:
            relevant: Set of relevant document IDs
            retrieved: Set of retrieved document IDs (top-k)
            k: Cutoff rank
        
        Returns:
            Precision value in [0, 1]
        """
        if not retrieved:
            return 0.0
        
        hits = len(relevant & retrieved)
        return hits / min(len(retrieved), k)


class MAPCalculator:
    """Calculate Mean Average Precision (MAP)."""
    
    @staticmethod
    def average_precision(relevant: Set, ranked_docs: List[Tuple]) -> float:
        """
        Average Precision = (1/|relevant|) * Σ P(k) * rel(k)
        where rel(k) is 1 if k-th document is relevant
        
        Args:
            relevant: Set of relevant document IDs
            ranked_docs: List of (doc_id, score) tuples in rank order
        
        Returns:
            AP value in [0, 1]
        """
        if not relevant:
            return 0.0
        
        score = 0.0
        hits = 0
        
        for k, (doc_id, _) in enumerate(ranked_docs, 1):
            if doc_id in relevant:
                hits += 1
                score += hits / k
        
        return score / len(relevant)
    
    @staticmethod
    def map_at_k(relevant_list: List[Set], ranked_docs_list: List[List[Tuple]], k: int = 10) -> float:
        """
        Mean Average Precision across multiple queries
        
        Args:
            relevant_list: List of relevant doc sets per query
            ranked_docs_list: List of ranked doc lists per query
            k: Cutoff rank
        
        Returns:
            MAP@K value
        """
        aps = []
        for relevant, ranked_docs in zip(relevant_list, ranked_docs_list):
            truncated = ranked_docs[:k]
            ap = MAPCalculator.average_precision(relevant, truncated)
            aps.append(ap)
        
        return np.mean(aps) if aps else 0.0


class NDCGCalculator:
    """Calculate Normalized Discounted Cumulative Gain (NDCG)."""
    
    @staticmethod
    def dcg_at_k(relevant: Set, ranked_docs: List[Tuple], k: int = 10) -> float:
        """
        DCG@K = Σ(rel(i) / log2(i+1)) for i in 1..k
        
        Args:
            relevant: Set of relevant document IDs
            ranked_docs: List of (doc_id, score) tuples
            k: Cutoff rank
        
        Returns:
            DCG value
        """
        dcg = 0.0
        
        for i, (doc_id, _) in enumerate(ranked_docs[:k], 1):
            if doc_id in relevant:
                dcg += 1.0 / np.log2(i + 1)
        
        return dcg
    
    @staticmethod
    def ideal_dcg_at_k(relevant: Set, k: int = 10) -> float:
        """
        Ideal DCG (IDCG) for k relevant documents
        
        Args:
            relevant: Set of relevant document IDs
            k: Cutoff rank
        
        Returns:
            Ideal DCG value
        """
        idcg = 0.0
        num_relevant = min(len(relevant), k)
        
        for i in range(1, num_relevant + 1):
            idcg += 1.0 / np.log2(i + 1)
        
        return idcg
    
    @staticmethod
    def ndcg_at_k(relevant: Set, ranked_docs: List[Tuple], k: int = 10) -> float:
        """
        NDCG@K = DCG@K / IDCG@K
        
        Args:
            relevant: Set of relevant document IDs
            ranked_docs: List of (doc_id, score) tuples
            k: Cutoff rank
        
        Returns:
            NDCG value in [0, 1]
        """
        dcg = NDCGCalculator.dcg_at_k(relevant, ranked_docs, k)
        idcg = NDCGCalculator.ideal_dcg_at_k(relevant, k)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg


class MRRCalculator:
    """Calculate Mean Reciprocal Rank (MRR)."""
    
    @staticmethod
    def reciprocal_rank(relevant: Set, ranked_docs: List[Tuple]) -> float:
        """
        Reciprocal Rank = 1 / (rank of first relevant document)
        
        Args:
            relevant: Set of relevant document IDs
            ranked_docs: List of (doc_id, score) tuples
        
        Returns:
            RR value
        """
        for i, (doc_id, _) in enumerate(ranked_docs, 1):
            if doc_id in relevant:
                return 1.0 / i
        
        return 0.0
    
    @staticmethod
    def mrr_at_k(relevant_list: List[Set], ranked_docs_list: List[List[Tuple]], k: int = 10) -> float:
        """
        Mean Reciprocal Rank across multiple queries
        
        Args:
            relevant_list: List of relevant doc sets per query
            ranked_docs_list: List of ranked doc lists per query
            k: Cutoff rank
        
        Returns:
            MRR@K value
        """
        rrs = []
        for relevant, ranked_docs in zip(relevant_list, ranked_docs_list):
            truncated = ranked_docs[:k]
            rr = MRRCalculator.reciprocal_rank(relevant, truncated)
            rrs.append(rr)
        
        return np.mean(rrs) if rrs else 0.0


# ============================================================================
# COMPREHENSIVE EVALUATOR
# ============================================================================

@dataclass
class MetricResult:
    """Result for a single metric."""
    metric_name: str
    k: int
    mean: float
    std: float
    min: float
    max: float
    count: int


class ComprehensiveEvaluator:
    """
    Unified evaluation interface for all metrics.
    
    Computes 10+ metrics at multiple K values [1, 5, 10, 20]
    """
    
    def __init__(self, k_values: List[int] = None, verbose: bool = True):
        """
        Initialize evaluator.
        
        Args:
            k_values: List of K values to evaluate at
            verbose: Print results
        """
        self.k_values = k_values or [1, 5, 10, 20]
        self.verbose = verbose
        self.results = {}
        
        # Metric calculators
        self.recall_calc = RecallCalculator()
        self.precision_calc = PrecisionCalculator()
        self.map_calc = MAPCalculator()
        self.ndcg_calc = NDCGCalculator()
        self.mrr_calc = MRRCalculator()
    
    def evaluate(
        self,
        qrels: Dict[str, Dict],
        rankings: Dict[str, List[Tuple]]
    ) -> Dict:
        """
        Evaluate full ranking results.
        
        Args:
            qrels: Dict of {query_id: {doc_id: relevance_score}}
            rankings: Dict of {query_id: [(doc_id, score), ...]}
        
        Returns:
            Dictionary of metric results
        """
        if self.verbose:
            logger.info(f"Evaluating {len(rankings)} queries with {len(self.k_values)} K values...")
        
        # Prepare data
        query_ids = sorted(qrels.keys())
        relevant_list = [set(qrels[qid].keys()) for qid in query_ids]
        ranked_docs_list = [rankings.get(qid, []) for qid in query_ids]
        
        results = {}
        
        # Evaluate each metric at each K
        for k in self.k_values:
            recall_scores = self._compute_metric(
                "Recall@K",
                query_ids,
                relevant_list,
                ranked_docs_list,
                k,
                self._calc_recall
            )
            
            precision_scores = self._compute_metric(
                "Precision@K",
                query_ids,
                relevant_list,
                ranked_docs_list,
                k,
                self._calc_precision
            )
            
            map_scores = self._compute_metric(
                "MAP@K",
                query_ids,
                relevant_list,
                ranked_docs_list,
                k,
                self._calc_map
            )
            
            ndcg_scores = self._compute_metric(
                "NDCG@K",
                query_ids,
                relevant_list,
                ranked_docs_list,
                k,
                self._calc_ndcg
            )
            
            mrr_scores = self._compute_metric(
                "MRR@K",
                query_ids,
                relevant_list,
                ranked_docs_list,
                k,
                self._calc_mrr
            )
            
            # Aggregate results
            for metric_name, scores in [
                ("Recall", recall_scores),
                ("Precision", precision_scores),
                ("MAP", map_scores),
                ("NDCG", ndcg_scores),
                ("MRR", mrr_scores)
            ]:
                key = f"{metric_name}@{k}"
                results[key] = {
                    "mean": float(np.mean(scores)),
                    "std": float(np.std(scores)),
                    "min": float(np.min(scores)),
                    "max": float(np.max(scores)),
                    "count": len(scores)
                }
        
        self.results = results
        
        if self.verbose:
            self._print_results(results)
        
        return results
    
    def _compute_metric(self, name, query_ids, relevant_list, ranked_docs_list, k, calc_fn):
        """Helper to compute metric for all queries."""
        scores = []
        for relevant, ranked_docs in zip(relevant_list, ranked_docs_list):
            score = calc_fn(relevant, ranked_docs, k)
            scores.append(score)
        return scores
    
    @staticmethod
    def _calc_recall(relevant, ranked_docs, k):
        """Calculate Recall@K."""
        retrieved = set(doc_id for doc_id, _ in ranked_docs[:k])
        return RecallCalculator.recall_at_k(relevant, retrieved, k)
    
    @staticmethod
    def _calc_precision(relevant, ranked_docs, k):
        """Calculate Precision@K."""
        retrieved = set(doc_id for doc_id, _ in ranked_docs[:k])
        return PrecisionCalculator.precision_at_k(relevant, retrieved, k)
    
    @staticmethod
    def _calc_map(relevant, ranked_docs, k):
        """Calculate MAP@K."""
        truncated = ranked_docs[:k]
        return MAPCalculator.average_precision(relevant, truncated)
    
    @staticmethod
    def _calc_ndcg(relevant, ranked_docs, k):
        """Calculate NDCG@K."""
        return NDCGCalculator.ndcg_at_k(relevant, ranked_docs, k)
    
    @staticmethod
    def _calc_mrr(relevant, ranked_docs, k):
        """Calculate MRR@K."""
        truncated = ranked_docs[:k]
        return MRRCalculator.reciprocal_rank(relevant, truncated)
    
    def _print_results(self, results: Dict) -> None:
        """Print formatted results."""
        logger.info("\n" + "="*70)
        logger.info("EVALUATION RESULTS")
        logger.info("="*70)
        
        for key, metrics in sorted(results.items()):
            logger.info(f"\n{key}")
            logger.info(f"  Mean:  {metrics['mean']:.4f}")
            logger.info(f"  Std:   {metrics['std']:.4f}")
            logger.info(f"  Min:   {metrics['min']:.4f}")
            logger.info(f"  Max:   {metrics['max']:.4f}")
    
    def get_summary(self) -> Dict:
        """Get summary of results (mean values only)."""
        return {
            key: metrics["mean"]
            for key, metrics in self.results.items()
        }
    
    def get_formatted_report(self) -> str:
        """Get formatted report as string."""
        if not self.results:
            return "No evaluation results available"
        
        report = "\n" + "="*70 + "\n"
        report += "SENTINEL EVALUATION REPORT\n"
        report += "="*70 + "\n\n"
        
        # Group by metric type
        metric_types = set()
        for key in self.results.keys():
            metric_types.add(key.split("@")[0])
        
        for metric in sorted(metric_types):
            report += f"\n{metric} RESULTS:\n"
            report += "-"*40 + "\n"
            
            for k in sorted(self.k_values):
                key = f"{metric}@{k}"
                if key not in self.results:
                    continue
                
                metrics = self.results[key]
                report += f"  @{k:2d}: {metrics['mean']:.4f} ± {metrics['std']:.4f} "
                report += f"[{metrics['min']:.4f}, {metrics['max']:.4f}]\n"
        
        report += "\n" + "="*70 + "\n"
        return report
