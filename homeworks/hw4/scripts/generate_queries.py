#!/usr/bin/env python3
"""
Synthetic Query Generation for HW4 RAG Evaluation

Generates realistic user queries that test complex cooking methods, appliance settings,
and specific ingredient techniques that are best answered by retrieving existing recipes.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import litellm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class QueryGenerator:
    """Generates synthetic queries for recipe retrieval evaluation."""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.generated_queries = []
    
    def extract_salient_facts(self, recipe: Dict[str, Any]) -> str:
        """
        Use LLM to extract salient facts from a recipe that would make good query targets.
        """
        
        # Prepare recipe text
        recipe_text = self._format_recipe_for_llm(recipe)
        
        prompt = f"""
Analyze this recipe and identify 1-2 specific, technical details that would be difficult to generate from scratch but are clearly answerable by this exact recipe. Focus on:

1. **Specific cooking techniques/methods** (e.g., "marinate for 4 hours", "bake at 375°F for exactly 25 minutes")
2. **Appliance settings** (e.g., "air fryer at 400°F for 12 minutes", "pressure cook for 8 minutes")  
3. **Ingredient preparation details** (e.g., "slice onions paper-thin", "whip cream to soft peaks")
4. **Timing specifics** (e.g., "rest dough for 30 minutes", "simmer for 45 minutes")
5. **Temperature precision** (e.g., "internal temp 165°F", "oil heated to 350°F")

Return the most distinctive fact(s) that someone might specifically search for:

Recipe:
{recipe_text}

Salient Fact(s):
"""
        
        try:
            response = litellm.completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error extracting facts: {e}")
            return ""
    
    def generate_realistic_query(self, recipe: Dict[str, Any], salient_fact: str) -> str:
        """
        Generate a realistic user query that would specifically need this recipe to answer.
        """
        
        recipe_name = recipe.get('name', 'Unknown Recipe')
        ingredients = ', '.join(recipe.get('ingredients', [])[:5])
        
        prompt = f"""
Create a realistic, specific user query that a home cook might ask, which can ONLY be answered well by this exact recipe. The query should:

1. Sound natural and conversational (like a real person asking)
2. Focus on the specific technical detail: "{salient_fact}"
3. Be challenging - requiring this exact recipe's information to answer properly
4. Avoid mentioning the recipe name directly

Context:
- Recipe: {recipe_name}
- Key ingredients: {ingredients}
- Salient fact: {salient_fact}

Examples of good query styles:
- "What temperature and time for air fryer frozen chicken tenders?"
- "How long should I marinate beef for Korean bulgogi?"
- "What's the exact oven temperature for crispy roasted vegetables?"
- "How do I get the right consistency for homemade pasta dough?"

Generate ONE specific query:
"""
        
        try:
            response = litellm.completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=100
            )
            return response.choices[0].message.content.strip().strip('"')
        except Exception as e:
            print(f"Error generating query: {e}")
            return ""
    
    def process_single_recipe(self, recipe: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single recipe to generate a query."""
        try:
            # Extract salient facts
            salient_fact = self.extract_salient_facts(recipe)
            
            if not salient_fact or len(salient_fact.strip()) < 10:
                return None
            
            # Generate query
            query = self.generate_realistic_query(recipe, salient_fact)
            
            if not query or len(query.strip()) < 10:
                return None
            
            return {
                "query": query,
                "salient_fact": salient_fact,
                "source_recipe_id": recipe['id'],
                "source_recipe_name": recipe['name'],
                "source_recipe_url": f"recipe_id_{recipe['id']}",  # Placeholder URL
                "ingredients": recipe.get('ingredients', []),
                "cooking_time": recipe.get('minutes', 0),
                "tags": recipe.get('tags', [])
            }
            
        except Exception as e:
            print(f"Error processing recipe {recipe.get('id', 'unknown')}: {e}")
            return None
    
    def _format_recipe_for_llm(self, recipe: Dict[str, Any]) -> str:
        """Format recipe for LLM processing."""
        formatted = f"**{recipe.get('name', 'Unknown')}**\n"
        
        if recipe.get('description'):
            formatted += f"Description: {recipe['description']}\n"
        
        if recipe.get('minutes'):
            formatted += f"Cooking time: {recipe['minutes']} minutes\n"
        
        if recipe.get('ingredients'):
            formatted += f"Ingredients: {', '.join(recipe['ingredients'])}\n"
        
        if recipe.get('steps'):
            formatted += "Instructions:\n"
            for i, step in enumerate(recipe['steps'], 1):
                formatted += f"{i}. {step}\n"
        
        return formatted
    
    def generate_queries_parallel(self, recipes: List[Dict[str, Any]], 
                                max_queries: int = 100, 
                                max_workers: int = 10) -> List[Dict[str, Any]]:
        """
        Generate queries in parallel using ThreadPoolExecutor.
        """
        print(f"Generating up to {max_queries} queries from {len(recipes)} recipes...")
        
        # Shuffle recipes to get diverse selection
        shuffled_recipes = recipes.copy()
        random.shuffle(shuffled_recipes)
        
        # Limit to max_queries
        target_recipes = shuffled_recipes[:max_queries * 2]  # Process more to account for failures
        
        queries = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_recipe = {
                executor.submit(self.process_single_recipe, recipe): recipe 
                for recipe in target_recipes
            }
            
            # Collect results with progress bar
            for future in tqdm(as_completed(future_to_recipe), 
                             total=len(future_to_recipe), 
                             desc="Processing recipes"):
                
                result = future.result()
                if result:
                    queries.append(result)
                    
                    # Stop if we have enough queries
                    if len(queries) >= max_queries:
                        break
        
        print(f"Successfully generated {len(queries)} queries")
        return queries[:max_queries]
    
    def save_queries(self, queries: List[Dict[str, Any]], output_path: Path) -> None:
        """Save generated queries to JSON file."""
        print(f"Saving {len(queries)} queries to {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump(queries, file, indent=2, ensure_ascii=False)
        
        print("Queries saved successfully")
    
    def print_sample_queries(self, queries: List[Dict[str, Any]], n_samples: int = 5) -> None:
        """Print sample queries for review."""
        print(f"\n--- Sample Generated Queries ({n_samples}) ---")
        
        for i, query_data in enumerate(queries[:n_samples]):
            print(f"\nQuery {i+1}:")
            print(f"  Query: '{query_data['query']}'")
            print(f"  Salient Fact: {query_data['salient_fact']}")
            print(f"  Source Recipe: {query_data['source_recipe_name']}")
            print(f"  Cooking Time: {query_data['cooking_time']} minutes")
            print(f"  Key Ingredients: {', '.join(query_data['ingredients'][:3])}...")


def load_processed_recipes(recipes_path: Path) -> List[Dict[str, Any]]:
    """Load processed recipes from JSON file."""
    print(f"Loading recipes from {recipes_path}")
    
    with open(recipes_path, 'r', encoding='utf-8') as file:
        recipes = json.load(file)
    
    print(f"Loaded {len(recipes)} recipes")
    return recipes


def filter_complex_recipes(recipes: List[Dict[str, Any]], min_steps: int = 5, min_ingredients: int = 4) -> List[Dict[str, Any]]:
    """Filter recipes that are complex enough to generate good queries."""
    filtered = []
    
    for recipe in recipes:
        # Filter criteria for good query generation
        has_enough_steps = recipe.get('n_steps', 0) >= min_steps
        has_enough_ingredients = recipe.get('n_ingredients', 0) >= min_ingredients
        has_description = bool(recipe.get('description', '').strip())
        has_cooking_time = recipe.get('minutes', 0) > 0
        
        # Look for recipes with specific cooking techniques
        steps_text = ' '.join(recipe.get('steps', [])).lower()
        ingredients_text = ' '.join(recipe.get('ingredients', [])).lower()
        
        # Prioritize recipes with specific techniques/equipment
        has_specific_techniques = any(term in steps_text for term in [
            'temperature', 'degrees', 'minutes', 'hours',
            'air fryer', 'oven', 'grill', 'pressure cook',
            'marinate', 'simmer', 'broil', 'sauté'
        ])
        
        if (has_enough_steps and has_enough_ingredients and 
            (has_description or has_cooking_time or has_specific_techniques)):
            filtered.append(recipe)
    
    print(f"Filtered to {len(filtered)} complex recipes")
    return filtered


def main():
    """Main query generation pipeline."""
    # Paths
    base_path = Path(__file__).parent.parent
    recipes_path = base_path / "data" / "processed_recipes.json"
    queries_path = base_path / "data" / "synthetic_queries.json"
    
    if not recipes_path.exists():
        print(f"Processed recipes not found: {recipes_path}")
        print("Run process_recipes.py first")
        return
    
    # Load recipes
    recipes = load_processed_recipes(recipes_path)
    
    # Filter for complex recipes that will generate good queries
    complex_recipes = filter_complex_recipes(recipes)
    
    if len(complex_recipes) < 100:
        print(f"Warning: Only {len(complex_recipes)} complex recipes found")
        print("Using all available recipes")
        complex_recipes = recipes
    
    # Generate queries
    generator = QueryGenerator()
    queries = generator.generate_queries_parallel(
        complex_recipes, 
        max_queries=200,  # Target 100+ queries
        max_workers=32    # Adjust based on API rate limits
    )
    
    if len(queries) == 0:
        print("No queries generated. Check API configuration.")
        return
    
    # Show samples
    generator.print_sample_queries(queries)
    
    # Save queries
    generator.save_queries(queries, queries_path)
    
    # Print summary
    print(f"\n--- Generation Summary ---")
    print(f"Processed {len(complex_recipes)} complex recipes")
    print(f"Generated {len(queries)} queries")
    print(f"Success rate: {len(queries)/len(complex_recipes)*100:.1f}%")
    
    # Analyze query characteristics
    avg_query_length = sum(len(q['query'].split()) for q in queries) / len(queries)
    cooking_times = [q['cooking_time'] for q in queries if q['cooking_time'] > 0]
    avg_cooking_time = sum(cooking_times) / len(cooking_times) if cooking_times else 0
    
    print(f"Average query length: {avg_query_length:.1f} words")
    print(f"Average recipe cooking time: {avg_cooking_time:.1f} minutes")


if __name__ == "__main__":
    main() 