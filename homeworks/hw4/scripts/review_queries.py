#!/usr/bin/env python3
"""
Manual Query Review Interface for HW4

Provides an interface to manually review, refine, and select the best
synthetic queries for the retrieval evaluation dataset.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional


class QueryReviewer:
    """Interactive interface for reviewing and refining synthetic queries."""
    
    def __init__(self):
        self.reviewed_queries = []
        self.refined_queries = []
        
    def load_queries(self, queries_path: Path) -> List[Dict[str, Any]]:
        """Load synthetic queries from JSON file."""
        print(f"Loading queries from {queries_path}")
        
        with open(queries_path, 'r', encoding='utf-8') as file:
            queries = json.load(file)
        
        print(f"Loaded {len(queries)} queries for review")
        return queries
    
    def display_query(self, query_data: Dict[str, Any], index: int) -> None:
        """Display a single query for review."""
        print(f"\n{'='*80}")
        print(f"QUERY #{index + 1}")
        print(f"{'='*80}")
        print(f"Query: '{query_data['query']}'")
        print(f"Target Recipe: {query_data['source_recipe_name']}")
        print(f"Salient Fact: {query_data['salient_fact']}")
        print(f"Cooking Time: {query_data['cooking_time']} minutes")
        print(f"Key Ingredients: {', '.join(query_data['ingredients'][:5])}")
        print(f"Recipe ID: {query_data['source_recipe_id']}")
        print("-" * 80)
    
    def get_user_feedback(self) -> str:
        """Get user feedback on the current query."""
        print("\nOptions:")
        print("  [k] Keep as-is")
        print("  [r] Refine/edit the query")
        print("  [s] Skip this query")
        print("  [q] Quit review session")
        print("  [h] Help/show options again")
        
        while True:
            choice = input("\nYour choice: ").strip().lower()
            if choice in ['k', 'r', 's', 'q', 'h']:
                return choice
            print("Invalid choice. Please enter k, r, s, q, or h.")
    
    def refine_query(self, original_query: str) -> Optional[str]:
        """Allow user to refine a query."""
        print(f"\nOriginal query: '{original_query}'")
        print("Enter your refined version (or press Enter to cancel):")
        
        refined = input("Refined query: ").strip()
        
        if not refined:
            print("Refinement cancelled.")
            return None
        
        print(f"Refined query: '{refined}'")
        confirm = input("Confirm this refinement? (y/n): ").strip().lower()
        
        if confirm == 'y':
            return refined
        else:
            print("Refinement not saved.")
            return None
    
    def review_queries_interactive(self, queries: List[Dict[str, Any]], 
                                 max_review: int = 20) -> List[Dict[str, Any]]:
        """Interactive review session for queries."""
        print(f"\n{'='*80}")
        print("INTERACTIVE QUERY REVIEW SESSION")
        print(f"{'='*80}")
        print(f"You will review up to {max_review} randomly selected queries.")
        print("For each query, you can keep it, refine it, or skip it.")
        print("Focus on realism, specificity, and answerability.")
        
        # Randomly sample queries for review
        review_queries = random.sample(queries, min(len(queries), max_review))
        
        approved_queries = []
        
        for i, query_data in enumerate(review_queries):
            self.display_query(query_data, i)
            
            while True:
                choice = self.get_user_feedback()
                
                if choice == 'h':
                    continue  # Show options again
                elif choice == 'q':
                    print("\nQuitting review session...")
                    return approved_queries
                elif choice == 's':
                    print("Skipping this query.")
                    break
                elif choice == 'k':
                    print("Keeping query as-is.")
                    approved_queries.append(query_data)
                    break
                elif choice == 'r':
                    refined_query = self.refine_query(query_data['query'])
                    if refined_query:
                        # Create refined version
                        refined_data = query_data.copy()
                        refined_data['query'] = refined_query
                        refined_data['refined'] = True
                        refined_data['original_query'] = query_data['query']
                        
                        approved_queries.append(refined_data)
                        print("Refined query added to approved list.")
                    break
        
        print(f"\nReview complete! Approved {len(approved_queries)} queries.")
        return approved_queries
    
    def batch_filter_queries(self, queries: List[Dict[str, Any]], 
                           criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply automatic filtering criteria to queries."""
        print("Applying automatic filters...")
        
        filtered = []
        
        for query in queries:
            # Length filters
            query_words = len(query['query'].split())
            if query_words < criteria.get('min_query_words', 4):
                continue
            if query_words > criteria.get('max_query_words', 20):
                continue
            
            # Content filters
            query_lower = query['query'].lower()
            
            # Skip overly generic queries
            generic_terms = ['recipe', 'how to make', 'how do i make']
            if any(term in query_lower for term in generic_terms):
                continue
            
            # Prefer queries with specific cooking terms
            specific_terms = [
                'temperature', 'degrees', 'minutes', 'hours',
                'air fryer', 'oven', 'pressure cook', 'grill',
                'marinate', 'simmer', 'sauté', 'broil',
                'setting', 'time', 'temp'
            ]
            
            if any(term in query_lower for term in specific_terms):
                filtered.append(query)
            elif len(query.get('salient_fact', '').split()) > 8:
                # Keep if salient fact is detailed
                filtered.append(query)
        
        print(f"Filtered to {len(filtered)} queries meeting criteria")
        return filtered
    
    def save_reviewed_queries(self, queries: List[Dict[str, Any]], output_path: Path) -> None:
        """Save reviewed and refined queries."""
        print(f"Saving {len(queries)} reviewed queries to {output_path}")
        
        # Add metadata
        output_data = {
            "metadata": {
                "total_queries": len(queries),
                "refined_count": len([q for q in queries if q.get('refined', False)]),
                "review_timestamp": str(Path(__file__).stat().st_mtime)
            },
            "queries": queries
        }
        
        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump(output_data, file, indent=2, ensure_ascii=False)
        
        print("Reviewed queries saved successfully")
    
    def print_review_summary(self, original_count: int, reviewed_queries: List[Dict[str, Any]]) -> None:
        """Print summary of review process."""
        refined_count = len([q for q in reviewed_queries if q.get('refined', False)])
        
        print(f"\n{'='*80}")
        print("REVIEW SUMMARY")
        print(f"{'='*80}")
        print(f"Original queries: {original_count}")
        print(f"Queries after review: {len(reviewed_queries)}")
        print(f"Queries refined: {refined_count}")
        print(f"Approval rate: {len(reviewed_queries)/original_count*100:.1f}%")
        
        if reviewed_queries:
            # Show some examples
            print(f"\n--- Sample Approved Queries ---")
            for i, query in enumerate(reviewed_queries[:3]):
                refined_marker = " (REFINED)" if query.get('refined', False) else ""
                print(f"{i+1}. '{query['query']}'{refined_marker}")


def main():
    """Main review interface."""
    # Paths
    base_path = Path(__file__).parent.parent
    synthetic_queries_path = base_path / "data" / "synthetic_queries.json"
    reviewed_queries_path = base_path / "data" / "evaluation_dataset.json"
    
    if not synthetic_queries_path.exists():
        print(f"Synthetic queries not found: {synthetic_queries_path}")
        print("Run generate_queries.py first")
        return
    
    # Initialize reviewer
    reviewer = QueryReviewer()
    
    # Load queries
    queries = reviewer.load_queries(synthetic_queries_path)
    
    if len(queries) == 0:
        print("No queries to review")
        return
    
    print(f"\nReview mode options:")
    print("1. Interactive review (manually review and refine queries)")
    print("2. Automatic filtering (apply criteria-based filtering)")
    print("3. Both (filter first, then interactive review)")
    
    mode = input("\nChoose review mode (1/2/3): ").strip()
    
    reviewed_queries = []
    
    if mode in ['2', '3']:
        # Apply automatic filtering
        filter_criteria = {
            'min_query_words': 5,
            'max_query_words': 15
        }
        filtered_queries = reviewer.batch_filter_queries(queries, filter_criteria)
        
        if mode == '2':
            reviewed_queries = filtered_queries
        else:  # mode == '3'
            # Continue with interactive review
            reviewed_queries = reviewer.review_queries_interactive(filtered_queries, max_review=10)
    
    elif mode == '1':
        # Only interactive review
        reviewed_queries = reviewer.review_queries_interactive(queries, max_review=15)
    
    else:
        print("Invalid mode selected")
        return
    
    if reviewed_queries:
        # Save results
        reviewer.save_reviewed_queries(reviewed_queries, reviewed_queries_path)
        
        # Print summary
        reviewer.print_review_summary(len(queries), reviewed_queries)
    else:
        print("No queries approved for evaluation dataset")


if __name__ == "__main__":
    main() 