#!/usr/bin/env python3
"""
Retrieval Evaluation for HW4 RAG System

Evaluates BM25 retrieval performance using synthetic queries and calculates
standard information retrieval metrics: Recall@1, Recall@3, Recall@5, Recall@10, and MRR.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any
import statistics

# Add backend to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "backend"))
from retrieval import create_retriever
from evaluation_utils import BaseRetrievalEvaluator, load_queries


class RetrievalEvaluator(BaseRetrievalEvaluator):
    """Extended evaluator with additional analysis for HW4."""
    
    def analyze_by_query_characteristics(self, results: List[Dict[str, Any]]):
        """Analyze performance by query characteristics."""
        print(f"\n--- Additional Analysis ---")
        
        # Analyze by query length
        short_queries = [r for r in results if len(r['original_query'].split()) <= 8]
        long_queries = [r for r in results if len(r['original_query'].split()) > 8]
        
        if short_queries and long_queries:
            short_recall = statistics.mean([r['recall_5'] for r in short_queries])
            long_recall = statistics.mean([r['recall_5'] for r in long_queries])
            print(f"Short queries (≤8 words) Recall@5: {short_recall:.3f} ({len(short_queries)} queries)")
            print(f"Long queries (>8 words) Recall@5: {long_recall:.3f} ({len(long_queries)} queries)")
        
        # Analyze by recipe complexity
        complex_recipes = [r for r in results if len(r['salient_fact'].split()) > 10]
        simple_recipes = [r for r in results if len(r['salient_fact'].split()) <= 10]
        
        if complex_recipes and simple_recipes:
            complex_recall = statistics.mean([r['recall_5'] for r in complex_recipes])
            simple_recall = statistics.mean([r['recall_5'] for r in simple_recipes])
            print(f"Complex salient facts Recall@5: {complex_recall:.3f} ({len(complex_recipes)} queries)")
            print(f"Simple salient facts Recall@5: {simple_recall:.3f} ({len(simple_recipes)} queries)")
    
    def print_final_summary(self, results: List[Dict[str, Any]]):
        """Print comprehensive final summary."""
        final_metrics = self.calculate_aggregate_metrics(results)
        print(f"\n{'='*80}")
        print("FINAL EVALUATION SUMMARY")
        print(f"{'='*80}")
        print(f"📊 Overall Performance:")
        print(f"   • Recall@1:  {final_metrics['recall_at_1']:.3f} ({final_metrics['recall_at_1']*100:.1f}%)")
        print(f"   • Recall@3:  {final_metrics['recall_at_3']:.3f} ({final_metrics['recall_at_3']*100:.1f}%)")
        print(f"   • Recall@5:  {final_metrics['recall_at_5']:.3f} ({final_metrics['recall_at_5']*100:.1f}%)")
        print(f"   • Recall@10: {final_metrics['recall_at_10']:.3f} ({final_metrics['recall_at_10']*100:.1f}%)")
        print(f"   • MRR:       {final_metrics['mean_reciprocal_rank']:.3f}")
        
        print(f"\n📈 Query Success:")
        print(f"   • Total queries evaluated: {final_metrics['total_queries']}")
        print(f"   • Target found (any rank): {final_metrics['queries_found']} ({final_metrics['queries_found']/final_metrics['total_queries']*100:.1f}%)")
        print(f"   • Target not found:        {final_metrics['queries_not_found']} ({final_metrics['queries_not_found']/final_metrics['total_queries']*100:.1f}%)")
        
        if final_metrics['average_rank_when_found']:
            print(f"\n🎯 Ranking Analysis:")
            print(f"   • Average rank when found: {final_metrics['average_rank_when_found']:.2f}")
            print(f"   • Median rank when found:  {final_metrics['median_rank_when_found']:.0f}")
        
        print(f"\n💡 Performance Insights:")
        success_rate = final_metrics['recall_at_5']
        if success_rate >= 0.7:
            print(f"   • Excellent retrieval performance (Recall@5 ≥ 70%)")
        elif success_rate >= 0.5:
            print(f"   • Good retrieval performance (Recall@5 ≥ 50%)")
        elif success_rate >= 0.3:
            print(f"   • Moderate retrieval performance (Recall@5 ≥ 30%)")
        else:
            print(f"   • Poor retrieval performance (Recall@5 < 30%)")
        
        mrr = final_metrics['mean_reciprocal_rank']
        if mrr >= 0.6:
            print(f"   • Excellent ranking quality (MRR ≥ 0.6)")
        elif mrr >= 0.4:
            print(f"   • Good ranking quality (MRR ≥ 0.4)")
        elif mrr >= 0.2:
            print(f"   • Moderate ranking quality (MRR ≥ 0.2)")
        else:
            print(f"   • Poor ranking quality (MRR < 0.2)")
        
        print(f"{'='*80}")


def main():
    """Main evaluation pipeline."""
    # Paths
    base_path = Path(__file__).parent.parent
    recipes_path = base_path / "data" / "processed_recipes.json"
    queries_path = base_path / "data" / "synthetic_queries.json"
    index_path = base_path / "data" / "bm25_index.pkl"
    results_path = base_path / "results" / "retrieval_evaluation.json"
    
    # Check required files
    if not recipes_path.exists():
        print(f"Processed recipes not found: {recipes_path}")
        print("Run process_recipes.py first")
        return
    
    if not queries_path.exists():
        print(f"Synthetic queries not found: {queries_path}")
        print("Run generate_queries.py first")
        return
    
    # Load data
    queries = load_queries(queries_path)
    
    if len(queries) == 0:
        print("No queries to evaluate")
        return
    
    # Create retriever
    print("Setting up BM25 retriever...")
    retriever = create_retriever(recipes_path, index_path)
    
    # Print retriever stats
    stats = retriever.get_stats()
    print(f"\nRetriever loaded: {stats['total_recipes']} recipes indexed")
    
    # Run evaluation
    evaluator = RetrievalEvaluator(retriever)
    results = evaluator.evaluate_all_queries(queries, top_k=5)
    
    # Print results
    evaluator.print_detailed_results(results, show_failures=True, max_examples=5)
    
    # Additional analysis
    evaluator.analyze_by_query_characteristics(results)
    
    # Save results
    evaluator.save_results(results, results_path, experiment_name="baseline_bm25")
    
    # Final summary
    evaluator.print_final_summary(results)


if __name__ == "__main__":
    main() 