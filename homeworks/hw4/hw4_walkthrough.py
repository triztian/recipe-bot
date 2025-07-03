import marimo

__generated_with = "0.12.8"
app = marimo.App(width="columns")


@app.cell(column=0, hide_code=True)
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import json
    from pathlib import Path
    import os
    import sys
    from typing import List, Dict, Tuple
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.graph_objects as go
    import plotly.express as px
    from IPython.display import Markdown, display
    return (
        Dict,
        List,
        Markdown,
        Path,
        Tuple,
        display,
        go,
        json,
        mo,
        np,
        os,
        pd,
        plt,
        px,
        sns,
        sys,
    )


@app.cell
def _(Path):
    BASE_PATH = Path('homeworks/hw4')
    return (BASE_PATH,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        # Homework 4: Recipe Bot Retrieval Evaluation Walkthrough

        Interactive walkthrough for:

        - Building a BM25 retrieval system for recipes
        - Generating synthetic queries for evaluation
        - Measuring retrieval performance with standard IR metrics
        - (Optional) Using LLM query rewriting to improve retrieval
        """
    )
    return


@app.cell(column=1, hide_code=True)
def _(mo):
    mo.md(
        """
        ## Part 1: Process Recipe Data

        - **Input**: `data/RAW_recipes.csv` (Kaggle dataset)
        - **Script**: `scripts/process_recipes.py`
        - **Output**: `data/processed_recipes.json`
        - **Purpose**: Turn raw dataset from kaggle into filtered and cleaned dataset ready to use
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Input""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        I downloaded as zip and placed the in the data folder manually. You can automate this with the kaggle python API, but I don't think it's neccesary for 1 time download tasks. Keep it simple!

        It has:

        - **Recipe names** - titles users can search for
        - **Ingredients lists** - what you need to make each dish
        - **Step-by-step instructions** - how to actually cook it
        - **Cooking times** - how long recipes take
        - **User tags** - categories like "easy", "vegetarian", "quick"
        - **Nutrition info** - calories, protein, etc.
        """
    )
    return


@app.cell
def _(BASE_PATH, pd):
    pd.read_csv(BASE_PATH/'data'/'RAW_recipes.csv')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Script""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        The raw data is processed by `scripts/process_recipes.py`.  This is a data cleaning step largely.

        - Remove extra whitspase.
        - Parses ingredient list, steps, and lists
        - Get longest recipes
        - Save in a standard format
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Ouptut""")
    return


@app.cell
def _(BASE_PATH, json):
    # Load processed recipes
    recipes = json.load(open(BASE_PATH/'data'/'processed_recipes.json', 'r'))
    return (recipes,)


@app.cell
def _(mo, recipes):
    # Create recipe browser widget
    recipe_index = mo.ui.slider(
        start=0, 
        stop=len(recipes)-1, 
        value=0, 
        label="Recipe Index",
        show_value=True
    )

    mo.md(f"""
    #### Interactive Recipe Browser

    {recipe_index}
    """)
    return (recipe_index,)


@app.cell
def _(mo, recipe_index, recipes):
    # Display selected recipe
    selected_recipe = recipes[recipe_index.value]

    mo.md(f"""
    **Recipe: {selected_recipe['name']}**

    - **ID**: {selected_recipe['id']}
    - **Cooking Time**: {selected_recipe['minutes']} minutes
    - **Number of Steps**: {selected_recipe['n_steps']}
    - **Number of Ingredients**: {selected_recipe['n_ingredients']}

    <details>
    <summary><b>Ingredients</b></summary>

    {'<br>'.join(f"- {ing}" for ing in selected_recipe['ingredients'])}

    </details>

    <details>
    <summary><b>Instructions</b></summary>

    {'<br>'.join(f"{i+1}. {step}" for i, step in enumerate(selected_recipe['steps']))}

    </details>

    <details>
    <summary><b>Tags</b></summary>

    {', '.join(selected_recipe['tags']) if selected_recipe['tags'] else 'No tags'}

    </details>
    """)
    return (selected_recipe,)


@app.cell(column=2, hide_code=True)
def _(mo):
    mo.md(
        """
        ## Part 2: Generate Synthetic Queries

        - **Input**: `data/processed_recipes.json`
        - **Script**: `scripts/generate_queries.py`
        - **Output**: `data/synthetic_queries.json` (queries with ground truth)
        - **Purpose**: LLM extracts facts → generates natural questions
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Script""")
    return


@app.cell
def _(mo):
    mo.md(r"""`scripts/generate_queries.py` generates synthetic queries in 2 steps""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Step 1
        Start by using extracting specific details from the query with this prompt
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Analyze this recipe and identify 1-2 specific, technical details that would be difficult to generate from scratch but are clearly answerable by this exact recipe. Focus on:

        1. **Specific cooking techniques/methods** (e.g., "marinate for 4 hours", "bake at 375°F for exactly 25 minutes")
        2. **Appliance settings** (e.g., "air fryer at 400°F for 12 minutes", "pressure cook for 8 minutes")  
        3. **Ingredient preparation details** (e.g., "slice onions paper-thin", "whip cream to soft peaks")
        4. **Timing specifics** (e.g., "rest dough for 30 minutes", "simmer for 45 minutes")
        5. **Temperature precision** (e.g., "internal temp 165°F", "oil heated to 350°F")

        Return the most distinctive fact(s) that someone might specifically search for:
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Step 2

        Then from the recipe and the facts that were extracted, create the synthetic querye
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
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
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Output""")
    return


@app.cell
def _(BASE_PATH, json, recipes):
    synthetic_queries = json.load(open(BASE_PATH/'data'/'synthetic_queries.json', 'r'))
    recipe_lookup = {r['id']: r for r in recipes}
    return recipe_lookup, synthetic_queries


@app.cell
def _(mo, synthetic_queries):
    # Create query selector
    query_selector = mo.ui.slider(
        start=0,
        stop=len(synthetic_queries)-1,
        value=0,
        label="Query Index",
        show_value=True
    )

    mo.md(f"""
    #### Browse Queries

    {query_selector}
    """)
    return (query_selector,)


@app.cell
def _(mo, query_selector, recipe_lookup, synthetic_queries):
    # Display selected query
    selected = synthetic_queries[query_selector.value]
    source_recipe = recipe_lookup.get(selected['source_recipe_id'])

    mo.md(f"""
    #### Query #{query_selector.value + 1}

    **🔍 Query Text:**
    > {selected['query']}

    **🎯 Target Recipe:** {selected['source_recipe_name']} (ID: {selected['source_recipe_id']})

    **⏱️ Cooking Time:** {selected['cooking_time']} minutes

    <br>**💡 Key Facts This Query Tests:**<br>

    {selected['salient_fact']}

    **🏷️ Recipe Tags:** {', '.join(selected['tags'][:10])}{'...' if len(selected['tags']) > 10 else ''}

    <details>
    <summary><b>📝 Full Recipe Details</b></summary>

    **Ingredients ({len(selected['ingredients'])}):**<br>
    {'<br>'.join(f"- {ing}" for ing in selected['ingredients'])}

    <br>**Steps ({source_recipe['n_steps'] if source_recipe else 'N/A'}):**<br>
    {'<br>'.join(f"{i+1}. {step}" for i, step in enumerate(source_recipe['steps']))}
    {'<br>...' if source_recipe and len(source_recipe['steps']) > 5 else ''}

    </details>
    """)
    return selected, source_recipe


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Part 3: Build BM25 Retrieval Engine

        - **Input**: Recipe texts from `processed_recipes.json`
        - **Script**: `backend/retrieval.py`
        - **Output**: BM25 search index (in memory)
        - **Function**: `retrieve_bm25(query, texts, index)`
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Script""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        To learn more check out some of my content on retrieval:

        - **BM25 Foundations**: [Here](https://www.kentro-learn.com/free-content/keyword-search-fundamentals) is a blog post and video discussion that covers the foundations of BM25 and how it works going from text processing, tokenization, TF-IDF, and up to BM25.

        - **Building a Practical Search MVP**:  [Here](https://isaacflath.com/blog/blog_post?fpath=posts%2F2025-03-17-Retrieval101.ipynb) is a blog post that steps through a full MVP for a search system incrementally explainin everything along the way.  This include a hybrid keyword + semantic search for initial ranking, then a cross-encoder as the reranking step.
        """
    )
    return


@app.cell(column=3, hide_code=True)
def _(mo):
    mo.md(
        """
        ## Part 4: Evaluate Retrieval Performance

        - **Input**: `synthetic_queries.json` + BM25 retrieval system
        - **Script**: `scripts/evaluate_retrieval.py`
        - **Output**: `results/retrieval_evaluation.json`
        - **Metrics**: Recall@k, MRR
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Script""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        1. Takes every query from synthetic_queries.json, searches for recipes using BM25, and checks if the correct recipe appears in the results
        2. Calculates metrics
        3. Some automated analysis (do this manually first!)

        **Key intuition:** This is measuring if your search engine can find the right recipe when given a natural language question. Recall measures where the result was found at all, and MRR measures how high up was the result in the ranking.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Output""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        #### Information Retrieval Metrics

        **Recall@k**: Fraction where target recipe is in top k results

        **MRR (Mean Reciprocal Rank)**: Average of 1/rank for each query

          - If target at rank 1: contributes 1.0
          - If target at rank 3: contributes 0.33
          - If not in top k: contributes 0
        """
    )
    return


@app.cell
def _(BASE_PATH, json):
    eval_results = json.load(open(BASE_PATH/'results'/'retrieval_evaluation.json', 'r'))
    eval_results
    return (eval_results,)


@app.cell
def _(eval_results, mo):
    metrics = eval_results['evaluation_summary']

    mo.md(f"""
    #### Retrieval Performance Results

    - **Recall@1**: {metrics['recall_at_1']:.3f} ({metrics['recall_at_1']*100:.1f}%)
    - **Recall@3**: {metrics['recall_at_3']:.3f} ({metrics['recall_at_3']*100:.1f}%)
    - **Recall@5**: {metrics['recall_at_5']:.3f} ({metrics['recall_at_5']*100:.1f}%)
    - **MRR**: {metrics['mrr']:.3f}
    """)
    return (metrics,)


@app.cell
def _(mo, synthetic_queries):
    # Create query selector
    query_selector2 = mo.ui.slider(
        start=0,
        stop=len(synthetic_queries)-1,
        value=0,
        label="Query Index",
        show_value=True
    )

    mo.md(f"""
    #### Browse Queries

    {query_selector2}
    """)
    return (query_selector2,)


@app.cell
def _(eval_results, mo, query_selector2):
    # Display selected query
    selected2 = eval_results['detailed_results'][query_selector2.value]

    mo.md(f"""
    **🔍 Query Text:**
    > {selected2['original_query']}

    {selected2['salient_fact']}

    **🎯 Target Recipe:** {selected2['target_recipe_name']} (ID: {selected2['target_recipe_id']})


    **🎯 Retrieved Recipes:** 

    {'<br>'.join(selected2['retrieved_names'])}


    """)
    return (selected2,)


@app.cell(column=4, hide_code=True)
def _(mo):
    mo.md(
        """
        ## Part 5: [Optional] Query Rewrite Agent

        - **Input**: Same as #4 + LLM rewriter
        - **Script**: `scripts/evaluate_retrieval_with_agent.py` and `query_rewrite_agent.py`
        - **Output**: `results/retrieval_comparison.json`
        - **Compares**: Baseline vs rewritten queries
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Script""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        Compares baseline vs enhanced retrieval - Measures improvement from the best rewriting strategy compared to raw BM25, tracking both successes (queries rescued) and failures (queries hurt)

        **Three Rewriting Strategies**

        1. **Keyword extraction**: Pull out key cooking terms
        2. **Query rewriting**: Restructure for better search
        3. **Query expansion**: Add synonyms and related terms

        **Key intuition:** This tests whether an LLM can rewrite user queries to better match how recipes are written.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Keyword Extraction
        ---
        You are a search optimization expert for a recipe database. Given a natural language cooking query, extract the most important keywords that would help find relevant recipes in a BM25 search.

        Focus on:

        1. **Cooking methods** (air fry, bake, grill, sauté, etc.)
        2. **Equipment/appliances** (air fryer, oven, pressure cooker, etc.)
        3. **Key ingredients** (chicken, vegetables, pasta, etc.)
        4. **Cooking specifics** (temperature, time, texture, etc.)
        5. **Food types** (appetizer, dessert, main dish, etc.)

        Remove filler words and focus on terms that would appear in recipe instructions or ingredients.

        Query: "{query}"

        Important search keywords (space-separated):
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Query Rewriting
        ---
        You are optimizing a cooking query for recipe search. Rewrite the query to be more effective for finding relevant recipes, focusing on terms that would appear in recipe titles, ingredients, and instructions.

        Guidelines:

        1. Use specific cooking terms instead of vague language
        2. Include equipment names if mentioned
        3. Add related cooking techniques
        4. Use ingredient names that commonly appear in recipes
        5. Keep it concise but descriptive
        6. Remove question words (what, how, when) and focus on content

        Original query: "{query}"

        Optimized search query:
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Synonym Expander
        ---
        Expand this cooking query by adding relevant synonyms and related cooking terms that might appear in recipes. This helps catch more relevant results in recipe search.

        Add terms that are:

        1. Synonyms for cooking methods mentioned
        2. Alternative ingredient names
        3. Related cooking techniques
        4. Equipment alternatives

        Keep the expansion focused and avoid unrelated terms.

        Original: "{query}"

        Expanded query with synonyms:
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Output""")
    return


@app.cell
def _(BASE_PATH, json):
    rewrite_results = json.load(open(BASE_PATH/'results'/'retrieval_comparison.json', 'r'))
    rewrite_results
    return (rewrite_results,)


@app.cell
def _(pd, rewrite_results):
    pd.DataFrame(rewrite_results['strategy_comparison'])
    return


@app.cell
def _(sys):
    sys.path.append('backend')
    from query_rewrite_agent import QueryRewriteAgent
    return (QueryRewriteAgent,)


@app.cell
def _(QueryRewriteAgent):
    rewriter = QueryRewriteAgent()
    return (rewriter,)


@app.cell
def _():
    query = "How do I make my cookies chewy instead of crunchy?"
    return (query,)


@app.cell(hide_code=True)
def _(mo, query, rewriter):
    mo.md(f"""
    > {query}

    **synonyms**

    {rewriter.expand_query_with_synonyms(query)}

    **Keywords**

    {rewriter.extract_search_keywords(query)}

    **Rewrite**

    {rewriter.rewrite_for_search(query)}
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
