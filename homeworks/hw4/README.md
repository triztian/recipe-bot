# Homework 4: Recipe Bot Retrieval Evaluation

## Dataset

This assignment uses recipe data from [Majumder et al. (2019)](https://aclanthology.org/D19-1613/) - "Generating Personalized Recipes from Historical User Preferences" (EMNLP-IJCNLP 2019). The dataset contains 180K+ recipes from Food.com with detailed ingredients, instructions, and user interactions.

**Data Options:**

1. **Quick Start**: Use our provided `data/processed_recipes.json` (200 longest recipes, pre-cleaned)
2. **Full Dataset**: Download the complete dataset from [Kaggle](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions) and place `RAW_recipes.csv` in `homeworks/hw4/data/`

The processed sample focuses on the longest recipes by text content to provide richer evaluation scenarios for retrieval testing.

## Your Task

Extend Recipe Bot with a RAG (Retrieval-Augmented Generation) component for handling specific queries about cooking methods, appliance settings, and ingredient variations. You'll build a complete evaluation pipeline to measure BM25 retrieval performance on complex, realistic cooking queries.

## Background

Traditional Recipe Bot focused on generating recipes for broad requests like "What should I make for dinner?". This extension handles more specific queries like:
- "What air fryer settings for frozen chicken tenders?"
- "How long to marinate beef for Korean bulgogi?"
- "What's the exact temperature for crispy roasted vegetables?"

## Assignment Parts

### Part 1: Create Your Retrieval Evaluation Dataset

**Goal**: Generate 100+ synthetic queries that test complex cooking scenarios.

#### Step 1: Process Recipe Data
Create `scripts/process_recipes.py`:
- Load and clean the provided `RAW_recipes.csv` (~5,000 recipes)
- Structure recipe data (ingredients, steps, tags, nutrition)
- Select the ~200 longest recipes by text content for richer evaluation
- Save as `data/processed_recipes.json`

#### Step 2: Build BM25 Retrieval Engine  
Create `backend/retrieval.py`:
- Implement BM25-based recipe search using `rank_bm25`
- Support index saving/loading for efficiency
- Provide `retrieve_bm25(query, corpus, top_n=5)` interface
- Handle recipe ranking and scoring

#### Step 3: Generate Synthetic Queries
Create `scripts/generate_queries.py`:
- Use LLM to generate realistic cooking queries
- Focus on complex scenarios requiring specific recipe knowledge
- Use ThreadPoolExecutor for parallel processing
- Generate 100+ queries with salient facts
- Save as `data/synthetic_queries.json`

### Part 2: Evaluate the BM25 Retriever

#### Step 4: Implement Evaluation
Create `scripts/evaluate_retrieval.py`:
- Load synthetic queries and retrieval engine
- For each query, run `retrieve_bm25()` and record results
- Calculate standard IR metrics:
  - **Recall@1**: Target recipe rank 1
  - **Recall@3**: Target recipe in top 3  
  - **Recall@5**: Target recipe in top 5
  - **MRR**: Mean Reciprocal Rank
- Save detailed results to `results/retrieval_evaluation.json`

#### Step 5: Manual Review (Optional)
Create `scripts/review_queries.py`:
- Interactive interface to manually review generated queries
- Refine queries for realism and challenge
- Export refined dataset for evaluation

### **Part 3: [OPTIONAL ADVANCED] Improve Retrieval with Query Rewrite Agent**
Implement an LLM-powered query rewrite agent to optimize queries before BM25 search and measure performance improvements.

## Implementation Steps

### Prerequisites
```bash
# Install dependencies
pip install rank-bm25 tqdm litellm python-dotenv

# Set up your LLM API key in .env file
echo "OPENAI_API_KEY=your_key_here" >> .env
```

### Step-by-Step Execution of Reference Implementation

```bash
cd homeworks/hw4

# 1. Process recipe data (creates processed_recipes.json)
python scripts/process_recipes.py

# 2. Generate synthetic queries (creates synthetic_queries.json)
python scripts/generate_queries.py

# 3. [Optional] Review and refine queries
python scripts/review_queries.py

# 4. Evaluate retrieval performance (creates retrieval_evaluation.json)
python scripts/evaluate_retrieval.py
```

## File Structure You'll Create

```
homeworks/hw4/
├── scripts/
│   ├── process_recipes.py          # Recipe data processing
│   ├── generate_queries.py         # Synthetic query generation  
│   ├── review_queries.py           # Manual query review (optional)
│   └── evaluate_retrieval.py       # Retrieval evaluation
├── data/
│   ├── RAW_recipes.csv             # Provided dataset
│   ├── processed_recipes.json      # Your cleaned recipe data
│   ├── synthetic_queries.json      # Your generated queries
│   └── evaluation_dataset.json     # Refined queries (if reviewed)
├── results/
│   └── retrieval_evaluation.json   # Your evaluation metrics
└── README.md                       # This file
```

## Query Generation Strategy

### Target Query Types
Focus on queries that require specific recipe knowledge:
1. **Appliance Settings**: "Air fryer temperature for crispy vegetables?"
2. **Timing Specifics**: "How long to marinate chicken for teriyaki?"
3. **Temperature Precision**: "What internal temp for medium-rare steak?"
4. **Technique Details**: "How to get crispy skin on roasted chicken?"

### LLM Prompting Approach
- **Step 1**: Extract salient facts from recipes (cooking methods, times, temperatures)
- **Step 2**: Generate realistic user queries that require those specific facts
- **Focus**: Complex, technical details that are hard to generate but easy to retrieve

## Evaluation Metrics

### Standard Information Retrieval Metrics
- **Recall@1**: Fraction of queries where target recipe is rank 1
- **Recall@3**: Fraction of queries where target recipe is in top 3
- **Recall@5**: Fraction of queries where target recipe is in top 5
- **MRR**: Mean Reciprocal Rank across all queries

### Expected Results
You should expect:
- **Recall@5**: 60-80% for well-formed queries
- **MRR**: 0.4-0.7 depending on query complexity
- **Higher performance** on technique-specific queries vs. general cooking questions

## Deliverables

1. **Working scripts** for all 4 components
2. **Evaluation results** with Recall@k and MRR metrics
3. **Brief analysis** (1-2 paragraphs) of:
   - What types of queries work well vs. poorly
   - How you would build an agent around this retriever
   - Ideas for improving retrieval performance

## Technical Requirements

### Dependencies
- `rank-bm25`: Fast BM25 implementation
- `litellm`: LLM integration for query generation
- `tqdm`: Progress bars for long-running operations
- `concurrent.futures`: Parallel processing support

### Performance Considerations
- Use ThreadPoolExecutor for parallel LLM calls
- Cache BM25 index for fast repeated searches
- Handle CSV parsing edge cases gracefully

## Part 3: [OPTIONAL ADVANCED] Query Rewrite Agent

**🎯 Learning Goal**: Understand how LLM agents can improve retrieval systems through query optimization.

### Overview
Natural language queries often don't match well with recipe text. For example:
- User: "What air fryer settings for frozen chicken tenders?"
- Better search: "air fryer frozen chicken tenders temperature time"

### Implementation Steps

#### Step 1: Build Query Rewrite Agent (`backend/query_rewrite_agent.py`)
Create an LLM-powered agent with three strategies:

1. **Keywords Extraction**: Extract key cooking terms and ingredients
2. **Query Rewriting**: Rewrite for better search effectiveness  
3. **Query Expansion**: Add synonyms and related cooking terms

**Performance Features:**
- **Parallel Processing**: Uses ThreadPoolExecutor for fast batch processing
- **Retry Logic**: Handles LLM API failures gracefully with exponential backoff
- **Progress Tracking**: Shows progress bars for long-running operations
- **Efficiency**: Processes 100+ queries with 3 strategies in seconds, not minutes

```python
class QueryRewriteAgent:
    def __init__(self, model: str = "gpt-4o-mini", max_workers: int = 10)
    def extract_search_keywords(self, query: str) -> str
    def rewrite_for_search(self, query: str) -> str  
    def expand_query_with_synonyms(self, query: str) -> str
    def batch_process_queries(self, queries: List[str], strategy: str) -> List[Dict]
    def batch_process_multiple_strategies(self, queries: List[str]) -> Dict[str, List[Dict]]
```

#### Step 2: Enhanced Evaluation (`scripts/evaluate_retrieval_with_agent.py`)
Compare baseline BM25 with agent-enhanced retrieval:

1. **Parallel Query Processing**: Pre-process all queries with all strategies simultaneously
2. **Efficient Evaluation**: Use pre-processed queries to avoid redundant LLM calls
3. **Performance Timing**: Track and report processing vs evaluation time
4. **Strategy Comparison**: Find best performing approach with detailed metrics
5. **Detailed Analysis**: Show where enhancement helps/hurts

**Key Performance Optimizations:**
- **Batch Processing**: Process all 100+ queries × 3 strategies in parallel
- **Pre-computation**: Separate query processing from evaluation for efficiency  
- **Progress Tracking**: Real-time progress bars and performance metrics
- **Error Handling**: Graceful degradation for failed LLM calls

#### Step 3: Performance Analysis
Measure improvements in:
- **Recall@5**: How often target recipe is found in top 5
- **MRR**: Quality of ranking when target is found
- **Query Rescue**: Failed queries that became successful
- **Query Degradation**: Successful queries that failed
- **Processing Speed**: Queries processed per second
- **Total Time**: Query processing + evaluation time

### Expected Results
Good implementations typically see:
- **5-15% improvement** in Recall@5 from query optimization
- **Keywords strategy**: Works well for technical queries
- **Rewrite strategy**: Best overall performance  
- **Expand strategy**: Helps with sparse matches
- **Processing Speed**: 50-100+ queries/second with parallel processing
- **Total Time**: Complete evaluation in under 30 seconds for 100 queries


## How to Run the Complete Pipeline

### Basic Implementation (Parts 1-2)
```bash
# From homeworks/hw4 directory
python scripts/process_recipes.py       # Process dataset
python scripts/generate_queries.py      # Generate synthetic queries  
python scripts/review_queries.py        # [Optional] Review and refine queries
python scripts/evaluate_retrieval.py    # Evaluate BM25 performance
```

### Advanced Implementation (Part 3 Optional)
```bash
# After completing basic implementation
python scripts/evaluate_retrieval_with_agent.py  # Compare with agent enhancement
```

The reference implementation provides all the above scripts as working examples.

## Reference Implementation

This repository contains a complete reference implementation showing one approach to this assignment. You can:
- **Study the code structure** to understand the RAG pipeline
- **Run the scripts yourself** to see expected behavior  
- **Implement your own version** from scratch for full learning value

The reference implementation includes both basic BM25 evaluation (Parts 1-2) and the optional query rewrite agent enhancement (Part 3).

### Reference Implementation Structure
```
homeworks/hw4/
├── scripts/
│   ├── process_recipes.py                  # Dataset processing
│   ├── generate_queries.py                 # Synthetic query generation  
│   ├── review_queries.py                   # Query review interface
│   ├── evaluate_retrieval.py               # BM25 evaluation (Parts 1-2)
│   └── evaluate_retrieval_with_agent.py    # Agent comparison (Part 3)
├── backend/
│   ├── retrieval.py                        # BM25 implementation
│   ├── query_rewrite_agent.py              # LLM query optimization (Part 3)
│   └── evaluation_utils.py                 # Reusable evaluation utilities
├── data/
│   ├── processed_recipes.json              # Processed recipe dataset
│   ├── synthetic_queries.json              # Generated evaluation queries
│   └── bm25_index.pkl                      # Saved BM25 index
└── results/
    ├── retrieval_evaluation.json           # Basic evaluation results
    ├── retrieval_baseline.json             # Baseline results (Part 3)
    ├── retrieval_enhanced.json             # Enhanced results (Part 3)
    └── retrieval_comparison.json           # Comparison analysis (Part 3)
```

---

Good luck building your retrieval evaluation system! 🍳📊 