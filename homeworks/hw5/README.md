# Homework 5 – Failure Transition Heat-Map

## Overview
Your cooking-assistant agent sometimes drops the spatula.  Every
conversation trace in this assignment contains **one failure**.  Your job is
pure analysis: given pre-labeled traces, build a transition matrix that shows
where the agent succeeds last and where it fails first, then visualize the
result as a heat-map and explain the patterns you see.

You do **not** need to call any LLMs or generate any additional data.  All
classification work has already been done for you.

---
## Data provided
`homeworks/hw5/data/labeled_traces.json` → list of 100 traces
```json
{
  "conversation_id": "a1b2…",
  "messages": [ {"role": "user", "content": "…"}, … ],
  "last_success_state": "GetRecipes",
  "first_failure_state": "GetWebInfo"
}
```
The two state fields form a **directed edge** that you will count in the
transition matrix.

If you are curious how the data were produced, see
`homeworks/hw5/generation/` (not part of the graded assignment).

---
## Pipeline state taxonomy
The agent's internal pipeline is abstracted to **10 canonical states**:

| # | State | Description |
|---|--------------------|-------------------------------------------|
| 1 | `ParseRequest`     | LLM interprets the user's message         |
| 2 | `PlanToolCalls`    | LLM decides which tools to invoke         |
| 3 | `GenCustomerArgs`  | LLM constructs arguments for customer DB  |
| 4 | `GetCustomerProfile` | Executes customer-profile tool         |
| 5 | `GenRecipeArgs`    | LLM constructs arguments for recipe DB    |
| 6 | `GetRecipes`       | Executes recipe-search tool               |
| 7 | `GenWebArgs`       | LLM constructs arguments for web search   |
| 8 | `GetWebInfo`       | Executes web-search tool                  |
| 9 | `ComposeResponse`  | LLM drafts the final answer               |
|10 | `DeliverResponse`  | Agent sends the answer                    |

Every trace succeeds through `last_success_state` and then fails at
`first_failure_state`.

---
## What you need to do
1. **Inspect the data**  
   Familiarize yourself with the JSON structure and the above state list.

2. **Build the transition matrix**  
   Count how many times each `(last_success → first_failure)` pair appears.

3. **Visualize**  
   Render a heat-map where rows = last-success, columns = first-failure.
   A starter script is provided:
   ```bash
   cd homeworks/hw5
   python analysis/transition_heatmaps.py
   ```
   This writes `results/failure_transition_heatmap.png`.

4. **Analyze**  
   • Which states fail most often?  
   • Do failures cluster around tool execution or argument generation?  
   • Any surprising low-frequency transitions?

5. **Deliverables**  
   • Heat-map PNG (commit to `homeworks/hw5/results/`).  
   • Short write-up (README or a separate markdown file) summarising your
     findings.

---
## File structure (after you generate the heat-map)
```
homeworks/hw5/
├── analysis/
│   └── transition_heatmaps.py   # you may tweak but it already works
├── data/
│   ├── labeled_traces.json      # provided
│   └── raw_traces.json          # provided for reference only
├── results/
│   └── failure_transition_heatmap.png  # ← your output
├── generation/  (ignore – instructor utilities)
└── README.md  # this file
```

---
## Advanced / optional
Curious how the dataset was made?  Peek inside `generation/` – it uses GPT-4.1
to pick failure states and author synthetic conversations.  Exploring or
modifying those scripts will **not** affect your grade.

Happy debugging 🛠️🍳
