import marimo

__generated_with = "0.12.8"
app = marimo.App(width="columns")


@app.cell(column=0)
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


@app.cell
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


@app.cell
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


@app.cell
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


@app.cell
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


@app.cell
def _(mo):
    mo.md(
        """
        ### Information Retrieval Metrics

        - **Recall@k**: Fraction where target recipe is in top k results
        - **MRR (Mean Reciprocal Rank)**: Average of 1/rank for each query
          - If target at rank 1: contributes 1.0
          - If target at rank 3: contributes 0.33
          - If not in top k: contributes 0
        """
    )
    return


@app.cell
def _(json, mo):
    # Load evaluation results
    try:
        with open('results/retrieval_evaluation.json', 'r') as f:
            eval_results = json.load(f)
        eval_loaded = True

        metrics = eval_results['metrics']

        mo.md(f"""
        ### Retrieval Performance Results

        - **Recall@1**: {metrics['recall_at_1']:.3f} ({metrics['recall_at_1']*100:.1f}%)
        - **Recall@3**: {metrics['recall_at_3']:.3f} ({metrics['recall_at_3']*100:.1f}%)
        - **Recall@5**: {metrics['recall_at_5']:.3f} ({metrics['recall_at_5']*100:.1f}%)
        - **MRR**: {metrics['mrr']:.3f}
        """)
    except FileNotFoundError:
        eval_results = None
        eval_loaded = False
        metrics = None
        mo.md("⚠️ **Evaluation results not found.** Run evaluateretrieval.pyevaluate_retrieval.py first.")
    return eval_loaded, eval_results, f, metrics


@app.cell
def _(eval_loaded, go, metrics, mo):
    if eval_loaded and metrics:
        # Create interactive recall curve
        k_values = [1, 3, 5]
        recall_values = [metrics['recall_at_1'], metrics['recall_at_3'], metrics['recall_at_5']]

        fig = go.Figure()

        # Add line trace
        fig.add_trace(go.Scatter(
            x=k_values,
            y=recall_values,
            mode='lines+markers+text',
            name='Recall@k',
            line=dict(color='blue', width=3),
            marker=dict(size=12),
            text=[f'{val:.3f}' for val in recall_values],
            textposition='top center'
        ))

        # Add ideal line
        fig.add_trace(go.Scatter(
            x=[1, 5],
            y=[1, 1],
            mode='lines',
            name='Perfect Retrieval',
            line=dict(color='green', width=2, dash='dash')
        ))

        fig.update_layout(
            title='BM25 Retrieval Performance',
            xaxis_title='k (Top-k results)',
            yaxis_title='Recall@k',
            yaxis_range=[0, 1.1],
            xaxis=dict(tickmode='array', tickvals=k_values),
            height=400,
            hovermode='x'
        )

        mo.ui.plotly(fig)
    else:
        mo.md("*Performance visualization will appear here*")
    return fig, k_values, recall_values


@app.cell
def _(eval_loaded, eval_results, mo):
    if eval_loaded and eval_results:
        # Analyze failures
        failed_queries = [r for r in eval_results['results'] if r['rank'] is None or r['rank'] > 5]

        mo.md(f"""
        ### Failure Analysis

        - **Total queries evaluated**: {len(eval_results['results'])}
        - **Failed queries (not in top 5)**: {len(failed_queries)} ({len(failed_queries)/len(eval_results['results'])*100:.1f}%)

        Common failure patterns:
        - **Vocabulary mismatch**: Query uses different terms than recipe
        - **Specificity**: Query too specific for BM25 keyword matching
        - **Context**: Query implies context not in recipe text
        """)
    else:
        failed_queries = []
    return (failed_queries,)


@app.cell
def _(eval_loaded, failed_queries, mo):
    if eval_loaded and failed_queries:
        # Create failure examples selector
        n_failures = min(len(failed_queries), 10)
        failure_index = mo.ui.slider(
            start=0,
            stop=n_failures-1,
            value=0,
            label="Browse Failed Queries",
            show_value=True
        )

        mo.md(f"""
        ### Failed Query Examples

        {failure_index}
        """)
    else:
        failure_index = None
    return failure_index, n_failures


@app.cell
def _(eval_loaded, failed_queries, failure_index, mo, recipes):
    if eval_loaded and failure_index is not None and failed_queries:
        failure = failed_queries[failure_index.value]

        failure_details = f"""
        **Failed Query #{failure_index.value + 1}**

        ❌ **Query**: "{failure['query']}"

        🎯 **Expected**: {recipes[failure['recipe_index']]['name']}
        """

        if failure['retrieved_indices']:
            failure_details += f"""

        📋 **Top 3 Retrieved Instead**:
        1. {recipes[failure['retrieved_indices'][0]]['name']}
        2. {recipes[failure['retrieved_indices'][1]]['name'] if len(failure['retrieved_indices']) > 1 else 'N/A'}
        3. {recipes[failure['retrieved_indices'][2]]['name'] if len(failure['retrieved_indices']) > 2 else 'N/A'}
        """

        mo.md(failure_details)
    return failure, failure_details


@app.cell(column=4, hide_code=True)
def _(mo):
    mo.md(
        """
        ## Part 5: [Optional] Query Rewrite Agent

        - **Input**: Same as #4 + LLM rewriter
        - **Script**: `scripts/evaluate_retrieval_with_agent.py`
        - **Output**: `results/retrieval_comparison.json`
        - **Compares**: Baseline vs rewritten queries
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        ### Motivation for Query Rewriting

        Natural language queries often don't match recipe text well:
        - User: "What's the secret to crispy fried chicken?"
        - Better for search: "crispy fried chicken coating technique temperature"

        ### Three Rewriting Strategies

        1. **Keyword extraction**: Pull out key cooking terms
        2. **Query rewriting**: Restructure for better search
        3. **Query expansion**: Add synonyms and related terms
        """
    )
    return


@app.cell
def _(json, mo):
    # Load comparison results if available
    try:
        with open('results/retrieval_comparison.json', 'r') as f:
            comparison = json.load(f)
        comparison_loaded = True

        mo.md("""
        ### Query Rewrite Performance Comparison
        """)
    except FileNotFoundError:
        comparison = None
        comparison_loaded = False
        mo.md("""
        ### Query Rewrite Results

        *No comparison results found. Run evaluateretrievalwitha≥nt.pyevaluate_retrieval_with_agent.py to see enhanced performance.*
        """)
    return comparison, comparison_loaded, f


@app.cell
def _(comparison, comparison_loaded, go, mo):
    if comparison_loaded and comparison:
        # Create comparison visualization
        strategies = ['baseline', 'keywords', 'rewrite', 'expand']
        metrics_to_show = ['recall_at_1', 'recall_at_3', 'recall_at_5', 'mrr']

        fig = go.Figure()

        # Add baseline
        baseline_metrics = comparison['baseline']['metrics']
        fig.add_trace(go.Bar(
            name='Baseline BM25',
            x=['Recall@1', 'Recall@3', 'Recall@5', 'MRR'],
            y=[baseline_metrics[metric] for metric in metrics_to_show],
            marker_color='lightblue'
        ))

        # Add best strategy
        best_strategy = comparison['best_strategy']
        best_metrics = comparison['enhanced'][best_strategy]['metrics']
        fig.add_trace(go.Bar(
            name=f'Enhanced ({best_strategy})',
            x=['Recall@1', 'Recall@3', 'Recall@5', 'MRR'],
            y=[best_metrics[metric] for metric in metrics_to_show],
            marker_color='darkgreen'
        ))

        fig.update_layout(
            title=f'Retrieval Performance: Baseline vs Best Strategy ({best_strategy})',
            yaxis_title='Score',
            yaxis_range=[0, 1],
            barmode='group',
            height=400
        )

        mo.ui.plotly(fig)
    return (
        baseline_metrics,
        best_metrics,
        best_strategy,
        fig,
        metrics_to_show,
        strategies,
    )


@app.cell
def _(comparison, comparison_loaded, mo):
    if comparison_loaded and comparison:
        improvement = comparison['best_improvement']
        best_strategy = comparison['best_strategy']

        mo.md(f"""
        ### Performance Summary

        - **Best Strategy**: {best_strategy}
        - **Overall Improvement**: {improvement:.1f}%
        - **Processing Time**: {comparison.get('processing_time', 'N/A')}

        The {best_strategy} strategy showed the best results, particularly for:
        - Queries with vocabulary mismatch
        - Technical cooking questions
        - Specific appliance or technique queries
        """)
    return best_strategy, improvement


@app.cell
def _(mo, query_selector):
    # Show transformations for selected query
    transformations = {
        "What's the secret to crispy fried chicken?": {
            "keywords": "crispy fried chicken secret",
            "rewritten": "crispy fried chicken coating technique temperature oil",
            "expanded": "crispy crunchy fried chicken breast thigh coating batter technique temperature deep fry oil"
        },
        "How long should I marinate steak?": {
            "keywords": "marinate steak time",
            "rewritten": "marinate steak hours time beef marinade",
            "expanded": "marinate marinade steak beef hours time overnight refrigerate seasoning"
        },
        "Air fryer settings for frozen french fries": {
            "keywords": "air fryer frozen french fries settings",
            "rewritten": "air fryer temperature time frozen french fries",
            "expanded": "air fryer temperature degrees time minutes frozen french fries potato crispy"
        },
        "Best way to caramelize onions": {
            "keywords": "caramelize onions best way",
            "rewritten": "caramelize onions technique heat time",
            "expanded": "caramelize caramelized onions brown sugar sweet technique low heat time pan"
        },
        "Temperature for baking sourdough bread": {
            "keywords": "temperature baking sourdough bread",
            "rewritten": "sourdough bread oven temperature degrees",
            "expanded": "sourdough bread baking temperature degrees fahrenheit oven time dutch oven steam"
        }
    }

    selected_transforms = transformations.get(query_selector.value, {})

    mo.md(f"""
    **Original**: {query_selector.value}

    **🔍 Keywords**: {selected_transforms.get('keywords', 'N/A')}

    **✏️ Rewritten**: {selected_transforms.get('rewritten', 'N/A')}

    **📚 Expanded**: {selected_transforms.get('expanded', 'N/A')}
    """)
    return selected_transforms, transformations


@app.cell
def _(mo):
    mo.md(
        """
        ## Key Takeaways

        ### 1. Retrieval is Critical for RAG
        - If relevant docs aren't retrieved, LLM can't use them
        - BM25 is simple but effective baseline
        - Measure with standard IR metrics

        ### 2. Synthetic Queries Enable Evaluation
        - Extract facts from documents
        - Generate queries that need those facts
        - Ensures ground truth for evaluation

        ### 3. Query-Document Mismatch is Common
        - Users ask questions differently than docs are written
        - Query rewriting can bridge this gap
        - But adds complexity and latency

        ### 4. Always Measure Baselines First
        - Simple BM25 often works surprisingly well
        - Understand failure modes before adding complexity
        - Incremental improvements compound
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## Practical Tips for Your Implementation

        - **Start simple**: Get basic BM25 working first
        - **Look at your data**: Examine queries that fail
        - **Parallel processing**: Use ThreadPoolExecutor for LLM calls
        - **Cache aggressively**: Save indices and results
        - **Test incrementally**: Verify each component works

        ### Next Steps

        1. Run processrecipes.pyprocess_recipes.py to prepare your data
        2. Implement BM25 retrieval in backendretrieval.pybackend/retrieval.py
        3. Generate synthetic queries with ≥≠ratequeries.pygenerate_queries.py
        4. Evaluate with evaluateretrieval.pyevaluate_retrieval.py
        5. (Optional) Try query rewriting for better performance
        """
    )
    return


if __name__ == "__main__":
    app.run()
