# Homework 5: Agent Failure Analysis with Transition Heatmaps

## Overview

This assignment focuses on analyzing agent failure patterns using synthetic conversation traces. You'll analyze pre-generated failure scenarios and create transition heatmaps to identify where failures occur in the agent pipeline.

## Learning Objectives

- **Pipeline State Analysis**: Understand how conversational agents move through different operational states
- **Failure Pattern Recognition**: Identify common transition points where agents fail
- **Transition Heatmaps**: Use data visualization to discover systematic issues in agent behavior
- **Tool-Specific Debugging**: Learn to identify which tools and operations are most failure-prone
- **Agent Debugging**: Apply systematic approaches to diagnosing agent problems

## Assignment Context

Modern conversational agents fail in predictable patterns that can be analyzed by examining their operational pipeline. By analyzing conversation traces and mapping them to agent states, we can identify exactly where failures occur and what causes them.

## Your Task

**Analyze pre-generated failure traces** and create transition heatmaps showing where failures occur in the agent pipeline. The synthetic traces have already been generated for you in `data/synthetic_traces.json`.

## Agent Pipeline States

You'll be working with these tool-specific agent states:

### Customer Database Operations
- **`FetchCustomer`** - Agent formulating query for customer data/preferences
- **`CustomerToolError`** - Customer database tool failures  
- **`ParseCustomerResults`** - Agent interpreting customer database results

### Recipe Search Operations  
- **`FetchRecipes`** - Agent formulating query for recipe search
- **`RecipeToolError`** - Recipe search tool failures
- **`ParseRecipeResults`** - Agent interpreting recipe search results

### Internet Search Operations
- **`FetchInternet`** - Agent formulating query for internet search
- **`InternetToolError`** - Internet search tool failures
- **`ParseInternetResults`** - Agent interpreting internet search results

### Dietary Restriction Operations
- **`FetchDietary`** - Agent formulating query for dietary restrictions
- **`DietaryToolError`** - Dietary restriction tool failures  
- **`ParseDietaryResults`** - Agent interpreting dietary restriction results

### Final Response Generation
- **`GenerateFinalResponse`** - Agent creating final response to user

## Assignment Parts

### Part 1: Analyze Conversation Traces

**Goal**: Use LLM-based classification to map conversation traces to agent pipeline states.

#### Step 1: Load and Examine Traces
- Examine the structure of `data/synthetic_traces.json`
- Understand the conversation format and failure indicators
- Identify the different types of failure modes present

#### Step 2: Implement State Classification
Create `analysis/transition_heatmaps.py`:
- **LLM-based Classification**: Use an LLM to analyze each failed conversation and identify where the first failure occurred
- **Parallel Processing**: Process multiple traces simultaneously for efficiency
- **State Mapping**: Map conversation content to specific pipeline states

#### Step 3: Build Failure Transition Matrix
- Count transitions where failures occur (e.g., `FetchRecipes` → `RecipeToolError`)
- Focus only on failed conversations
- Create a matrix showing failure counts between states

### Part 2: Generate Transition Heatmap

#### Step 4: Create Visualization
- Generate a heatmap showing "Failure Occurred In State →" 
- X-axis: "To State", Y-axis: "From State"
- Use failure counts as heatmap values
- Make the visualization clear and interpretable

#### Step 5: Analysis and Insights
- Identify the most common failure transitions
- Determine which tools are most failure-prone
- Analyze whether failures occur during tool execution or result processing
- Generate insights about agent reliability

## Implementation Approach

### LLM State Classification
Use an LLM to analyze each conversation and return just the failure transition:

```python
# Example LLM prompt:
"""
Analyze this conversation and identify WHERE THE FIRST FAILURE occurred.

Return ONLY two states: FromState,ToState

Examples:
- If recipe search failed: FetchRecipes,RecipeToolError
- If processing results failed: ParseRecipeResults,RecipeToolError
- If customer fetch failed: FetchCustomer,CustomerToolError
"""
```

### Expected Results
Your heatmap should show patterns like:
- **High tool execution failures**: `FetchX` → `XToolError`
- **High processing failures**: `ParseXResults` → `XToolError`
- **Recovery failures**: `XToolError` → `GenerateFinalResponse`
- **Tool-specific patterns**: Which tools (customer, recipe, internet, dietary) fail most often

## File Structure

```
homeworks/hw5/
├── README.md                           # This file
├── data/
│   ├── synthetic_traces.json          # Pre-generated failure traces (PROVIDED)
│   ├── failure_modes.json             # Failure mode definitions (PROVIDED)
│   └── customer_personas.json         # Customer personas (PROVIDED)
├── analysis/
│   └── transition_heatmaps.py         # YOUR IMPLEMENTATION
├── results/
│   ├── failure_transition_heatmap.png # Generated heatmap
│   └── transition_analysis_results.json # Analysis data
└── requirements.txt                   # Dependencies
```

## Getting Started

1. **Examine the data**: Look at `data/synthetic_traces.json` to understand the conversation format
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Set up API key**: Add your OpenAI API key to `.env` file
4. **Implement analysis**: Create the transition analysis in `analysis/transition_heatmaps.py`
5. **Generate heatmap**: Run your analysis to create the failure transition heatmap

## Reference Solution

Here's an example of what your failure transition heatmap should look like:

![Failure Transition Heatmap](https://github.com/user-attachments/assets/example-heatmap.png)

This reference heatmap shows:
- **FetchRecipes → RecipeToolError**: 51 failures (most common - recipe query formulation issues)
- **FetchCustomer → CustomerToolError**: 12 failures (customer query problems)  
- **FetchInternet → InternetToolError**: 12 failures (internet search query issues)
- **FetchRecipes → ParseRecipeResults**: 6 failures (recipe results interpretation problems)
- **FetchInternet → GenerateFinalResponse**: 1 failure (unusual transition pattern)

Key insights from this example:
- **Recipe tool has the most failures** - both in query formulation (51) and result processing (6)
- **Query formulation failures dominate** - most errors happen in the "Fetch" states
- **Customer and Internet tools** have moderate failure rates
- **Very few failures in Parse states** - agents are better at interpreting results than forming queries

Your analysis should produce similar insights about which tools fail most often and whether failures occur during query formulation or result interpretation.

## Key Insights to Look For

- **Tool Reliability**: Which tools (`customer`, `recipe`, `internet`, `dietary`) fail most often?
- **Failure Stages**: Do failures happen more during tool execution or result processing?
- **Cascading Failures**: Do certain failures lead to other failures?
- **Recovery Patterns**: How well does the agent recover from different types of failures?

Good luck analyzing agent failure patterns! 🤖📊 