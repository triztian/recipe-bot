#!/usr/bin/env python3
"""
Enhanced Retrieval Evaluation with Query Rewrite Agent

Compares baseline BM25 with query rewrite agent enhanced retrieval
to measure the impact of query optimization on retrieval performance.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any
import time

# Add backend to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "backend"))
from retrieval import create_retriever
from query_rewrite_agent import QueryRewriteAgent
from evaluation_utils import (
    BaseRetrievalEvaluator, 
    load_queries, 
    compare_retrieval_systems,
    print_comparison_results
)


class AgentRetrievalEvaluator(BaseRetrievalEvaluator):
    """Enhanced evaluator that uses pre-processed queries for efficiency."""
    
    def __init__(self, retriever, processed_queries: Dict[str, List[Dict[str, str]]] = None):
        """
        Initialize with pre-processed queries for all strategies.
        
        Args:
            retriever: BM25 retriever object
            processed_queries: Dict mapping strategy names to processed query lists
        """
        super().__init__(retriever, None)  # No query_processor needed
        self.processed_queries = processed_queries or {}
    
    def evaluate_all_queries_with_strategy(self, queries: List[Dict[str, Any]], 
                                         strategy: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Evaluate all queries using pre-processed queries for a specific strategy."""
        
        if strategy not in self.processed_queries:
            raise ValueError(f"Strategy '{strategy}' not found in processed queries")
        
        processed_query_list = self.processed_queries[strategy]
        
        if len(processed_query_list) != len(queries):
            raise ValueError(f"Mismatch: {len(queries)} original queries vs {len(processed_query_list)} processed")
        
        print(f"Evaluating {len(queries)} queries with {strategy} strategy...")
        
        results = []
        for i, (query_data, processed_query_data) in enumerate(zip(queries, processed_query_list)):
            # Use the processed query for search
            search_query = processed_query_data['processed_query']
            
            # Retrieve results
            max_k = max(top_k, 10)
            retrieval_results = self.retriever.retrieve_bm25(search_query, top_k=max_k)
            retrieved_ids = [recipe['id'] for recipe in retrieval_results]
            
            # Calculate metrics
            target_recipe_id = query_data['source_recipe_id']
            recall_1 = self.calculate_recall_at_k(retrieved_ids, target_recipe_id, 1)
            recall_3 = self.calculate_recall_at_k(retrieved_ids, target_recipe_id, 3)
            recall_5 = self.calculate_recall_at_k(retrieved_ids, target_recipe_id, 5)
            recall_10 = self.calculate_recall_at_k(retrieved_ids, target_recipe_id, 10)
            reciprocal_rank = self.calculate_reciprocal_rank(retrieved_ids, target_recipe_id)
            
            # Find actual rank
            target_rank = None
            if target_recipe_id in retrieved_ids:
                target_rank = retrieved_ids.index(target_recipe_id) + 1
            
            evaluation_result = {
                "original_query": query_data['query'],
                "search_query": search_query,
                "processing_strategy": strategy,
                "target_recipe_id": target_recipe_id,
                "target_recipe_name": query_data['source_recipe_name'],
                "salient_fact": query_data['salient_fact'],
                "retrieved_ids": retrieved_ids[:top_k],
                "retrieved_names": [recipe['name'] for recipe in retrieval_results[:top_k]],
                "target_rank": target_rank,
                "recall_1": recall_1,
                "recall_3": recall_3,
                "recall_5": recall_5,
                "recall_10": recall_10,
                "reciprocal_rank": reciprocal_rank,
                "bm25_scores": [recipe.get('bm25_score', 0.0) for recipe in retrieval_results[:top_k]]
            }
            
            results.append(evaluation_result)
        
        return results


def main():
    """Main evaluation pipeline comparing baseline and enhanced retrieval."""
    print("=" * 80)
    print("ENHANCED RETRIEVAL EVALUATION WITH QUERY REWRITE AGENT")
    print("=" * 80)
    
    # Paths
    base_path = Path(__file__).parent.parent
    recipes_path = base_path / "data" / "processed_recipes.json"
    queries_path = base_path / "data" / "synthetic_queries.json"
    index_path = base_path / "data" / "bm25_index.pkl"
    results_baseline_path = base_path / "results" / "retrieval_baseline.json"
    results_enhanced_path = base_path / "results" / "retrieval_enhanced.json"
    comparison_path = base_path / "results" / "retrieval_comparison.json"
    
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
    print(f"Retriever loaded: {stats['total_recipes']} recipes indexed")
    
    # Initialize query rewrite agent
    print("Initializing query rewrite agent...")
    agent = QueryRewriteAgent(max_workers=8)  # Use 8 workers for good parallelism
    
    print(f"\nEvaluating with {len(queries)} queries...")
    
    # === BASELINE EVALUATION ===
    print(f"\n{'='*60}")
    print("1. BASELINE BM25 RETRIEVAL")
    print(f"{'='*60}")
    
    baseline_evaluator = BaseRetrievalEvaluator(retriever)
    baseline_results = baseline_evaluator.evaluate_all_queries(queries, top_k=5)
    
    baseline_evaluator.print_detailed_results(baseline_results, show_failures=True, max_examples=3)
    baseline_evaluator.save_results(baseline_results, results_baseline_path, experiment_name="baseline_bm25")
    
    baseline_metrics = baseline_evaluator.calculate_aggregate_metrics(baseline_results)
    print(f"\n📊 Baseline Summary:")
    print(f"   Recall@5: {baseline_metrics['recall_at_5']:.3f}")
    print(f"   MRR:      {baseline_metrics['mean_reciprocal_rank']:.3f}")
    
    # === ENHANCED EVALUATION ===
    print(f"\n{'='*60}")
    print("2. ENHANCED RETRIEVAL WITH QUERY REWRITE AGENT")
    print(f"{'='*60}")
    
    # Step 1: Pre-process all queries with all strategies in parallel
    print("Pre-processing queries with all strategies in parallel...")
    query_strings = [q['query'] for q in queries]
    
    start_time = time.time()
    processed_queries = agent.batch_process_multiple_strategies(query_strings)
    processing_time = time.time() - start_time
    
    print(f"✅ Query processing completed in {processing_time:.2f} seconds")
    print(f"📊 Processing speed: {len(query_strings) * 3 / processing_time:.1f} queries/second")
    
    # Step 2: Evaluate each strategy using pre-processed queries
    strategies = ["keywords", "rewrite", "expand"]
    strategy_results = {}
    
    # Create enhanced evaluator with pre-processed queries
    enhanced_evaluator = AgentRetrievalEvaluator(retriever, processed_queries)
    
    for strategy in strategies:
        if strategy not in processed_queries or not processed_queries[strategy]:
            print(f"⚠️  Skipping {strategy} strategy - no processed queries")
            continue
            
        print(f"\n--- Evaluating {strategy.upper()} Strategy ---")
        
        start_time = time.time()
        results = enhanced_evaluator.evaluate_all_queries_with_strategy(queries, strategy, top_k=5)
        eval_time = time.time() - start_time
        
        metrics = enhanced_evaluator.calculate_aggregate_metrics(results)
        
        print(f"Results with {strategy} strategy (evaluated in {eval_time:.2f}s):")
        print(f"   Recall@5: {metrics['recall_at_5']:.3f} ({metrics['recall_at_5']*100:.1f}%)")
        print(f"   MRR:      {metrics['mean_reciprocal_rank']:.3f}")
        
        strategy_results[strategy] = {
            'results': results,
            'metrics': metrics,
            'processing_time': processing_time / len(strategies),  # Approximate
            'evaluation_time': eval_time
        }
    
    if not strategy_results:
        print("❌ No strategies completed successfully")
        return
    
    # Find best strategy
    best_strategy = max(strategy_results.keys(), 
                       key=lambda s: strategy_results[s]['metrics']['recall_at_5'])
    
    print(f"\n🏆 Best performing strategy: {best_strategy.upper()}")
    
    enhanced_results = strategy_results[best_strategy]['results']
    
    # Show detailed results for best strategy
    enhanced_evaluator.print_detailed_results(enhanced_results, show_failures=True, max_examples=3)
    enhanced_evaluator.save_results(enhanced_results, results_enhanced_path, 
                                  experiment_name=f"enhanced_{best_strategy}")
    
    # === COMPARISON ===
    print(f"\n{'='*60}")
    print("3. BASELINE vs ENHANCED COMPARISON")
    print(f"{'='*60}")
    
    comparison = compare_retrieval_systems(baseline_results, enhanced_results)
    print_comparison_results(comparison)
    
    # === PERFORMANCE ANALYSIS ===
    print(f"\n{'='*60}")
    print("4. PERFORMANCE ANALYSIS")
    print(f"{'='*60}")
    
    total_processing_time = processing_time
    total_evaluation_time = sum(s['evaluation_time'] for s in strategy_results.values())
    
    print(f"⏱️  Performance Timing:")
    print(f"   Query processing (all strategies): {total_processing_time:.2f}s")
    print(f"   Evaluation (all strategies): {total_evaluation_time:.2f}s")
    print(f"   Total time: {total_processing_time + total_evaluation_time:.2f}s")
    print(f"   Parallel efficiency: {len(queries) * 3 / total_processing_time:.1f} queries/second")
    
    # Show strategy comparison
    print(f"\n📊 Strategy Performance Comparison:")
    for strategy, data in strategy_results.items():
        metrics = data['metrics']
        print(f"   {strategy.upper()}:")
        print(f"     Recall@5: {metrics['recall_at_5']:.3f}")
        print(f"     MRR:      {metrics['mean_reciprocal_rank']:.3f}")
        print(f"     Eval time: {data['evaluation_time']:.2f}s")
    
    # Save comparison results
    comparison_data = {
        "baseline_experiment": "baseline_bm25",
        "enhanced_experiment": f"enhanced_{best_strategy}",
        "best_strategy": best_strategy,
        "performance_timing": {
            "query_processing_time": total_processing_time,
            "total_evaluation_time": total_evaluation_time,
            "queries_per_second": len(queries) * 3 / total_processing_time
        },
        "strategy_comparison": {
            strategy: {
                "recall_at_5": data['metrics']['recall_at_5'],
                "mrr": data['metrics']['mean_reciprocal_rank'],
                "evaluation_time": data['evaluation_time']
            }
            for strategy, data in strategy_results.items()
        },
        "detailed_comparison": comparison
    }
    
    print(f"\nSaving comparison results to {comparison_path}")
    with open(comparison_path, 'w', encoding='utf-8') as file:
        json.dump(comparison_data, file, indent=2, ensure_ascii=False)
    
    # === DETAILED ANALYSIS ===
    print(f"\n{'='*60}")
    print("5. DETAILED ANALYSIS")
    print(f"{'='*60}")
    
    # Show examples where enhancement helped
    print(f"\n--- Examples Where Query Rewriting Helped ---")
    helped_count = 0
    for baseline_result, enhanced_result in zip(baseline_results, enhanced_results):
        if (baseline_result['recall_5'] == 0.0 and enhanced_result['recall_5'] == 1.0):
            helped_count += 1
            if helped_count <= 3:  # Show first 3 examples
                print(f"\n{helped_count}. Original: '{baseline_result['original_query']}'")
                print(f"   Enhanced: '{enhanced_result['search_query']}' ({enhanced_result['processing_strategy']})")
                print(f"   Target: {enhanced_result['target_recipe_name']}")
                print(f"   Result: ❌ → ✅ (Found at rank {enhanced_result['target_rank']})")
    
    if helped_count == 0:
        print("   No clear examples where rewriting rescued failed queries.")
    else:
        print(f"   Total queries rescued: {helped_count}")
    
    # Show examples where enhancement hurt
    print(f"\n--- Examples Where Query Rewriting Hurt ---")
    hurt_count = 0
    for baseline_result, enhanced_result in zip(baseline_results, enhanced_results):
        if (baseline_result['recall_5'] == 1.0 and enhanced_result['recall_5'] == 0.0):
            hurt_count += 1
            if hurt_count <= 3:  # Show first 3 examples
                print(f"\n{hurt_count}. Original: '{baseline_result['original_query']}'")
                print(f"   Enhanced: '{enhanced_result['search_query']}' ({enhanced_result['processing_strategy']})")
                print(f"   Target: {enhanced_result['target_recipe_name']}")
                print(f"   Result: ✅ → ❌ (Lost from top 5)")
    
    if hurt_count == 0:
        print("   No examples where rewriting hurt performance.")
    else:
        print(f"   Total queries hurt: {hurt_count}")
    
    # Final recommendations
    print(f"\n{'='*60}")
    print("6. RECOMMENDATIONS")
    print(f"{'='*60}")
    
    improvement = comparison['improvements']['recall_at_5']['relative_improvement_pct']
    
    if improvement > 10:
        print("✅ STRONG RECOMMENDATION: Deploy query rewrite agent")
        print(f"   • {improvement:.1f}% improvement in Recall@5")
        print(f"   • Best strategy: {best_strategy}")
        print(f"   • Processing overhead: {total_processing_time:.1f}s for {len(queries)} queries")
    elif improvement > 5:
        print("⚠️  MODERATE RECOMMENDATION: Consider query rewrite agent")
        print(f"   • {improvement:.1f}% improvement in Recall@5")
        print(f"   • Monitor performance in production")
        print(f"   • Processing time: {total_processing_time:.1f}s")
    elif improvement > 0:
        print("💡 WEAK RECOMMENDATION: Query rewriting shows promise")
        print(f"   • {improvement:.1f}% improvement in Recall@5")
        print(f"   • Consider further tuning strategies")
    else:
        print("❌ NOT RECOMMENDED: No clear benefit from query rewriting")
        print(f"   • {improvement:.1f}% change in Recall@5")
        print(f"   • Stick with baseline BM25")
    
    print(f"\n✨ Evaluation complete! Check results in:")
    print(f"   • Baseline: {results_baseline_path}")
    print(f"   • Enhanced: {results_enhanced_path}")
    print(f"   • Comparison: {comparison_path}")


if __name__ == "__main__":
    main() 