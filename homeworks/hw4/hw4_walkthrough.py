import marimo

__generated_with = "0.12.8"
app = marimo.App(width="medium")


@app.cell
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


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Part 0: Download the Dataset

        I downloaded as zip and placed the rawrecipes.csvraw_recipes.csv in the datadata folder.  You can automate this with the kaggle python API, but I don't think it's neccesary for 1 time download tasks.  Keep it simple!
        """
    )
    return


@app.cell
def _():
    return


@app.cell
def _(BASE_PATH, pd):
    pd.read_csv(BASE_PATH/'data'/'RAW_recipes.csv')
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        What's in the recipe dataset that's useful for a chatbot:

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
def _(mo):
    mo.md("""## Part 1: Process Recipe Data""")
    return


@app.cell
def _(BASE_PATH, json, mo):
    # Load processed recipes
    recipes = json.load(open(BASE_PATH/'data'/'processed_recipes.json', 'r'))

    mo.md(f"""
    **Loaded {len(recipes)} recipes from the dataset**

    Let's explore the data interactively:
    """)
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
    ### Interactive Recipe Browser

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

    {chr(10).join(f"- {ing}" for ing in selected_recipe['ingredients'])}

    </details>

    <details>
    <summary><b>Instructions</b></summary>

    {chr(10).join(f"{i+1}. {step}" for i, step in enumerate(selected_recipe['steps']))}

    </details>

    <details>
    <summary><b>Tags</b></summary>

    {', '.join(selected_recipe['tags']) if selected_recipe['tags'] else 'No tags'}

    </details>
    """)
    return (selected_recipe,)


@app.cell
def _(mo):
    mo.md(
        """
        ### Recipe Processing Pipeline

        Key steps in processrecipes.pyprocess_recipes.py:
        - Parse CSV with proper handling of nested lists
        - Clean malformed data (nutrition, ingredients)
        - Select longest recipes by combined text length
        - Create searchable text representation
        """
    )
    return


@app.cell
def _(mo, pd, recipes):
    # Create recipe statistics
    recipe_stats = pd.DataFrame([{
        'name': r['name'],
        'text_length': len(r['name'] + ' '.join(r['ingredients']) + ' '.join(r['steps'])),
        'n_ingredients': r['n_ingredients'],
        'n_steps': r['n_steps'],
        'minutes': r['minutes']
    } for r in recipes])

    mo.md("""
    ### Recipe Dataset Statistics

    Distribution of recipe characteristics:
    """)
    return (recipe_stats,)


@app.cell
def _(mo, px, recipe_stats):
    # Create interactive visualizations
    fig1 = px.histogram(
        recipe_stats, 
        x='text_length', 
        nbins=30,
        title='Distribution of Recipe Text Length',
        labels={'text_length': 'Combined Text Length (characters)', 'count': 'Number of Recipes'}
    )
    fig1.update_layout(height=400)

    fig2 = px.scatter(
        recipe_stats,
        x='n_ingredients',
        y='n_steps',
        size='minutes',
        hover_data=['name'],
        title='Recipe Complexity: Ingredients vs Steps',
        labels={'n_ingredients': 'Number of Ingredients', 'n_steps': 'Number of Steps'}
    )
    fig2.update_layout(height=400)

    mo.md(f"""
    {mo.ui.plotly(fig1)}

    {mo.ui.plotly(fig2)}
    """)
    return fig1, fig2


@app.cell
def _(mo):
    mo.md(
        """
        ## Part 2: Build BM25 Retrieval Engine

        ### What is BM25?

        - **Best Match 25**: Classic information retrieval algorithm
        - Ranks documents by relevance to query terms
        - Considers:
          - Term frequency (how often query words appear)
          - Inverse document frequency (rarity of terms)
          - Document length normalization
        """
    )
    return


@app.cell
def _(recipes):
    # Create searchable text function
    def create_recipe_text(recipe):
        """Create searchable text from recipe components"""
        parts = [
            recipe['name'],
            ' '.join(recipe['ingredients']),
            ' '.join(recipe['steps']),
            ' '.join(recipe['tags'])
        ]
        return ' '.join(parts)

    # Create recipe texts
    recipe_texts = [create_recipe_text(r) for r in recipes]
    return create_recipe_text, recipe_texts


@app.cell
def _(mo, recipe_texts):
    # Import retrieval functions
    try:
        from retrieval import retrieve_bm25, build_bm25_index

        # Build BM25 index
        bm25_index = build_bm25_index(recipe_texts)
        retrieval_available = True

        mo.md("""
        ✅ **BM25 index built successfully**

        Let's test the retrieval with an interactive query:
        """)
    except ImportError:
        retrieval_available = False
        bm25_index = None
        retrieve_bm25 = None

        mo.md("""
        ⚠️ **Retrieval module not available**

        The retrieval functionality requires the backend module. Showing demo mode instead.
        """)
    return bm25_index, build_bm25_index, retrieval_available, retrieve_bm25


@app.cell
def _(mo, retrieval_available):
    # Create query input widget
    if retrieval_available:
        query_input = mo.ui.text(
            placeholder="Enter your search query (e.g., 'air fryer chicken wings crispy')",
            label="Search Query",
            full_width=True
        )

        top_n_select = mo.ui.select(
            options=[1, 3, 5, 10],
            value=5,
            label="Number of results"
        )

        mo.md(f"""
        ### Interactive Recipe Search

        {query_input}

        {top_n_select}
        """)
    else:
        query_input = None
        top_n_select = None
        mo.md("### Recipe Search (Demo Mode)")
    return query_input, top_n_select


@app.cell
def _(
    bm25_index,
    mo,
    query_input,
    recipe_texts,
    recipes,
    retrieval_available,
    retrieve_bm25,
    top_n_select,
):
    # Perform search and display results
    if retrieval_available and query_input and query_input.value:
        results = retrieve_bm25(
            query_input.value, 
            recipe_texts, 
            bm25_index, 
            top_n=top_n_select.value
        )

        results_html = f"<h4>Search Results for: '{query_input.value}'</h4>"
        for i, (idx, score) in enumerate(results):
            recipe = recipes[idx]
            results_html += f"""
            <div style='border: 1px solid #ddd; padding: 10px; margin: 10px 0; border-radius: 5px;'>
                <h5>{i+1}. {recipe['name']}</h5>
                <p><b>Score:</b> {score:.3f} | <b>Time:</b> {recipe['minutes']} min | 
                   <b>Ingredients:</b> {recipe['n_ingredients']} | <b>Steps:</b> {recipe['n_steps']}</p>
                <p><i>{recipe_texts[idx][:200]}...</i></p>
            </div>
            """

        mo.Html(results_html)
    else:
        mo.md("*Enter a query above to search recipes*")
    return i, idx, recipe, results, results_html, score


@app.cell
def _(mo):
    mo.md(
        """
        ## Part 3: Generate Synthetic Queries

        ### Query Generation Strategy

        Two-step process:
        1. **Extract salient facts** from recipes
           - Cooking methods and temperatures
           - Specific timings
           - Key techniques
        2. **Generate realistic queries** that would need those facts
           - Natural language questions
           - Focus on specificity
        """
    )
    return


@app.cell
def _(json, mo):
    # Load synthetic queries
    try:
        with open('data/synthetic_queries.json', 'r') as f:
            synthetic_queries = json.load(f)
        queries_loaded = True

        mo.md(f"""
        **Loaded {len(synthetic_queries)} synthetic queries**

        Let's explore them interactively:
        """)
    except FileNotFoundError:
        synthetic_queries = []
        queries_loaded = False
        mo.md("⚠️ **Synthetic queries not found.** Run ≥≠ratequeries.pygenerate_queries.py first.")
    return f, queries_loaded, synthetic_queries


@app.cell
def _(mo, queries_loaded, synthetic_queries):
    if queries_loaded and synthetic_queries:
        # Query browser
        query_index = mo.ui.slider(
            start=0,
            stop=min(len(synthetic_queries)-1, 99),
            value=0,
            label="Query Index",
            show_value=True
        )

        mo.md(f"""
        ### Query Browser

        {query_index}
        """)
    else:
        query_index = None
    return (query_index,)


@app.cell
def _(json, mo, queries_loaded, query_index, recipes, synthetic_queries):
    if queries_loaded and query_index:
        selected_query = synthetic_queries[query_index.value]
        target_recipe = recipes[selected_query['recipe_index']]

        mo.md(f"""
        **Query #{query_index.value + 1}**

        📝 **Query**: "{selected_query['query']}"

        🎯 **Target Recipe**: {target_recipe['name']}

        💡 **Salient Facts**:
        {chr(10).join(f"- {fact}" for fact in selected_query['salient_facts'])}

        <details>
        <summary><b>Target Recipe Details</b></summary>

        {json.dumps(target_recipe, indent=2)[:500]}...

        </details>
        """)
    return selected_query, target_recipe


@app.cell
def _(mo, queries_loaded, synthetic_queries):
    if queries_loaded and synthetic_queries:
        # Analyze query types
        query_categories = {
            'temperature': 0,
            'timing': 0,
            'technique': 0,
            'appliance': 0,
            'ingredient': 0,
            'other': 0
        }

        for q in synthetic_queries[:100]:
            query_lower = q['query'].lower()
            categorized = False

            if any(word in query_lower for word in ['temperature', 'degrees', 'heat', '°']):
                query_categories['temperature'] += 1
                categorized = True
            elif any(word in query_lower for word in ['how long', 'minutes', 'hours', 'time']):
                query_categories['timing'] += 1
                categorized = True
            elif any(word in query_lower for word in ['how to', 'technique', 'method', 'way to']):
                query_categories['technique'] += 1
                categorized = True
            elif any(word in query_lower for word in ['air fryer', 'instant pot', 'oven', 'grill', 'slow cooker']):
                query_categories['appliance'] += 1
                categorized = True
            elif any(word in query_lower for word in ['substitute', 'ingredient', 'replace']):
                query_categories['ingredient'] += 1
                categorized = True

            if not categorized:
                query_categories['other'] += 1

        mo.md("### Query Type Distribution")
    else:
        query_categories = {}
    return categorized, q, query_categories, query_lower


@app.cell
def _(mo, px, queries_loaded, query_categories):
    if queries_loaded and query_categories:
        # Create interactive pie chart
        fig = px.pie(
            values=list(query_categories.values()),
            names=list(query_categories.keys()),
            title='Distribution of Query Types',
            hole=0.4
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400)

        mo.ui.plotly(fig)
    else:
        mo.md("*Query type distribution will appear here*")
    return (fig,)


@app.cell
def _(mo):
    mo.md(
        """
        ## Part 4: Evaluate Retrieval Performance

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


@app.cell
def _(mo):
    mo.md(
        """
        ## Part 5: [Optional] Query Rewrite Agent

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
def _(mo):
    # Example transformations widget
    example_queries = [
        "What's the secret to crispy fried chicken?",
        "How long should I marinate steak?",
        "Air fryer settings for frozen french fries",
        "Best way to caramelize onions",
        "Temperature for baking sourdough bread"
    ]

    query_selector = mo.ui.select(
        options=example_queries,
        value=example_queries[0],
        label="Select Example Query"
    )

    mo.md(f"""
    ### Query Transformation Examples

    {query_selector}
    """)
    return example_queries, query_selector


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
