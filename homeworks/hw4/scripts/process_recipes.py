#!/usr/bin/env python3
"""
Recipe Data Processor for HW4 RAG Evaluation

Processes the RAW_recipes.csv file and creates clean, structured recipe data
for use in the BM25 retrieval system.
"""

import csv
import json
import ast
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from tqdm import tqdm


def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    if not text or text == "":
        return ""
    
    # Remove extra whitespace and normalize
    cleaned = re.sub(r'\s+', ' ', str(text).strip())
    return cleaned


def parse_list_string(list_str: str) -> List[str]:
    """Safely parse string representation of lists."""
    if not list_str or list_str == "":
        return []
    
    try:
        # Try to parse as literal list
        parsed = ast.literal_eval(list_str)
        if isinstance(parsed, list):
            return [clean_text(item) for item in parsed if item]
        else:
            return [clean_text(str(parsed))]
    except (ValueError, SyntaxError):
        # Fallback: split by common delimiters
        if ',' in list_str:
            return [clean_text(item.strip("'\"")) for item in list_str.split(',') if item.strip()]
        else:
            return [clean_text(list_str)]


def parse_nutrition(nutrition_str: str) -> Dict[str, float]:
    """Parse nutrition information."""
    if not nutrition_str:
        return {}
    
    try:
        nutrition_list = ast.literal_eval(nutrition_str)
        if len(nutrition_list) >= 7:
            return {
                "calories": float(nutrition_list[0]) if nutrition_list[0] else 0.0,
                "total_fat": float(nutrition_list[1]) if nutrition_list[1] else 0.0,
                "sugar": float(nutrition_list[2]) if nutrition_list[2] else 0.0,
                "sodium": float(nutrition_list[3]) if nutrition_list[3] else 0.0,
                "protein": float(nutrition_list[4]) if nutrition_list[4] else 0.0,
                "saturated_fat": float(nutrition_list[5]) if nutrition_list[5] else 0.0,
                "carbohydrates": float(nutrition_list[6]) if nutrition_list[6] else 0.0,
            }
    except (ValueError, IndexError, TypeError):
        pass
    
    return {}


def process_recipe(row: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """Process a single recipe row into structured format."""
    try:
        # Parse basic fields
        recipe_id = int(row.get('id', 0))
        name = clean_text(row.get('name', ''))
        description = clean_text(row.get('description', ''))
        
        if not name:  # Skip recipes without names
            return None
        
        # Parse time and step count
        minutes = int(row.get('minutes', 0)) if row.get('minutes') else 0
        n_steps = int(row.get('n_steps', 0)) if row.get('n_steps') else 0
        n_ingredients = int(row.get('n_ingredients', 0)) if row.get('n_ingredients') else 0
        
        # Parse lists
        ingredients = parse_list_string(row.get('ingredients', ''))
        steps = parse_list_string(row.get('steps', ''))
        tags = parse_list_string(row.get('tags', ''))
        
        # Parse nutrition
        nutrition = parse_nutrition(row.get('nutrition', ''))
        
        # Create full text for search indexing
        full_text_parts = [
            name,
            description,
            ' '.join(ingredients),
            ' '.join(steps),
            ' '.join(tags)
        ]
        full_text = ' '.join(filter(None, full_text_parts))
        
        recipe = {
            "id": recipe_id,
            "name": name,
            "description": description,
            "minutes": minutes,
            "ingredients": ingredients,
            "n_ingredients": n_ingredients,
            "steps": steps,
            "n_steps": n_steps,
            "tags": tags,
            "nutrition": nutrition,
            "submitted": row.get('submitted', ''),
            "contributor_id": int(row.get('contributor_id', 0)) if row.get('contributor_id') else 0,
            "full_text": full_text
        }
        
        return recipe
        
    except Exception as e:
        print(f"Error processing recipe: {e}")
        return None


def load_and_process_recipes(csv_path: Path, max_recipes: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load and process recipes from CSV file."""
    recipes = []
    
    print(f"Loading recipes from {csv_path}")
    
    with open(csv_path, 'r', encoding='utf-8') as file:
        # Use csv.DictReader to handle complex CSV parsing
        reader = csv.DictReader(file)
        
        for i, row in enumerate(tqdm(reader, desc="Processing recipes")):
            if max_recipes and i >= max_recipes:
                break
                
            recipe = process_recipe(row)
            if recipe:
                recipes.append(recipe)
    
    print(f"Successfully processed {len(recipes)} recipes")
    return recipes


def save_processed_recipes(recipes: List[Dict[str, Any]], output_path: Path) -> None:
    """Save processed recipes to JSON file."""
    print(f"Saving processed recipes to {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(recipes, file, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(recipes)} recipes")


def print_sample_recipes(recipes: List[Dict[str, Any]], n_samples: int = 3) -> None:
    """Print sample recipes for verification."""
    print(f"\n--- Sample Processed Recipes ({n_samples}) ---")
    
    for i, recipe in enumerate(recipes[:n_samples]):
        print(f"\nRecipe {i+1}:")
        print(f"  ID: {recipe['id']}")
        print(f"  Name: {recipe['name']}")
        print(f"  Description: {recipe['description'][:100]}...")
        print(f"  Ingredients ({len(recipe['ingredients'])}): {recipe['ingredients'][:3]}...")
        print(f"  Steps ({len(recipe['steps'])}): {recipe['steps'][0][:50] if recipe['steps'] else 'None'}...")
        print(f"  Tags: {recipe['tags'][:5]}...")
        print(f"  Minutes: {recipe['minutes']}")
        if recipe['nutrition']:
            print(f"  Calories: {recipe['nutrition'].get('calories', 'N/A')}")


def calculate_recipe_length_score(recipe: Dict[str, Any]) -> float:
    """Calculate a length score for ranking recipes by complexity/length."""
    # Simply use the length of the full text
    return len(recipe.get('full_text', ''))


def select_longest_recipes(recipes: List[Dict[str, Any]], top_n: int = 200) -> List[Dict[str, Any]]:
    """Select the top N longest recipes by full text length."""
    print(f"Selecting top {top_n} longest recipes (by text length) from {len(recipes)} total recipes...")
    
    # Sort recipes by full_text length (descending) and take top N
    sorted_recipes = sorted(recipes, key=lambda r: len(r.get('full_text', '')), reverse=True)
    longest_recipes = sorted_recipes[:top_n]
    
    print(f"Selected {len(longest_recipes)} longest recipes")
    
    # Print some stats about selected recipes
    if longest_recipes:
        text_lengths = [len(r.get('full_text', '')) for r in longest_recipes]
        avg_steps = sum(r.get('n_steps', 0) for r in longest_recipes) / len(longest_recipes)
        avg_ingredients = sum(r.get('n_ingredients', 0) for r in longest_recipes) / len(longest_recipes)
        avg_minutes = sum(r.get('minutes', 0) for r in longest_recipes if r.get('minutes', 0) > 0) / len([r for r in longest_recipes if r.get('minutes', 0) > 0])
        
        print(f"Selected recipes stats:")
        print(f"  Text length range: {min(text_lengths)} - {max(text_lengths)} characters")
        print(f"  Average text length: {sum(text_lengths) / len(text_lengths):.0f} characters")
        print(f"  Average steps: {avg_steps:.1f}")
        print(f"  Average ingredients: {avg_ingredients:.1f}")
        print(f"  Average cooking time: {avg_minutes:.1f} minutes")
    
    return longest_recipes


def main():
    """Main processing pipeline."""
    # Paths
    base_path = Path(__file__).parent.parent
    csv_path = base_path / "data" / "RAW_recipes.csv"
    output_path = base_path / "data" / "processed_recipes.json"
    
    # Process ALL recipes first (remove the limit to get full dataset)
    print("Loading and processing all recipes to find the longest ones...")
    all_recipes = load_and_process_recipes(csv_path, max_recipes=5000)
    
    # Select the 200 longest recipes
    longest_recipes = select_longest_recipes(all_recipes, top_n=200)
    
    # Show samples from the longest recipes
    print_sample_recipes(longest_recipes, n_samples=3)
    
    # Save the longest recipes (not all recipes)
    save_processed_recipes(longest_recipes, output_path)
    
    # Print summary statistics for the selected recipes
    print(f"\n--- Processing Summary (Top 200 Longest Recipes) ---")
    print(f"Total recipes processed: {len(all_recipes)}")
    print(f"Longest recipes selected: {len(longest_recipes)}")
    print(f"Average ingredients per recipe: {sum(r['n_ingredients'] for r in longest_recipes) / len(longest_recipes):.1f}")
    print(f"Average steps per recipe: {sum(r['n_steps'] for r in longest_recipes) / len(longest_recipes):.1f}")
    print(f"Average cooking time: {sum(r['minutes'] for r in longest_recipes if r['minutes'] > 0) / len([r for r in longest_recipes if r['minutes'] > 0]):.1f} minutes")
    
    # Show the complexity range
    if longest_recipes:
        text_lengths = [len(r.get('full_text', '')) for r in longest_recipes]
        print(f"Text length range: {min(text_lengths)} - {max(text_lengths)} characters")
        print(f"Average text length: {sum(text_lengths) / len(text_lengths):.0f} characters")


if __name__ == "__main__":
    main() 