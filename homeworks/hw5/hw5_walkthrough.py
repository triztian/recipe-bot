import marimo

__generated_with = "0.14.10"
app = marimo.App(
    width="medium",
    layout_file="layouts/hw5_walkthrough.grid.json",
)


@app.cell
def _():
    import marimo as mo
    import json, os
    from pathlib import Path
    from collections import Counter
    import pandas as pd
    return Counter, Path, json, mo, pd


@app.cell
def _(Path):
    BASE_PATH = Path('homeworks/hw5/')
    return (BASE_PATH,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Homework 5

    **Purpose**: Analyze agent failure patterns
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 1. JSON structure and State List""")
    return


@app.cell
def _(BASE_PATH, json):
    labeled_traces = json.load(open(BASE_PATH/'data'/'labeled_traces.json', 'r'))
    type(labeled_traces), len(labeled_traces)
    return (labeled_traces,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    What we have

    - A list of ~ 100 traces
    - Each trace has a `conversation_id`  we can refer to
    - We have all the messages and tool call information as well as what happened
    - We have the last success and first failure state, so we can see the trainsition
    """
    )
    return


@app.cell
def _(labeled_traces):
    labeled_traces[0]
    return


@app.cell
def _(labeled_traces, mo):
    trace_index_slider = mo.ui.slider(
        start=0,
        stop=len(labeled_traces) - 1,
        step=1,
        value=0,
        label="Trace Index"
    )
    trace_index_slider
    return (trace_index_slider,)


@app.cell
def _():
    import re
    def camel_to_regular(camel_string):
      s = re.sub(r'(?<!^)(?=[A-Z])', ' ', camel_string)
      return s.lower().title()
    return (camel_to_regular,)


@app.cell(hide_code=True)
def _(camel_to_regular, labeled_traces, mo, trace_index_slider):
    _trace = labeled_traces[trace_index_slider.value]

    # Create a list to hold message elements
    message_elements = []

    # Add the header information
    message_elements.append(mo.md(f"""
    ID: {_trace['conversation_id']}

    **Failure Transition:** {camel_to_regular(_trace['last_success_state'])} -> {camel_to_regular(_trace['first_failure_state'])}

    **Messages**
    """))

    # Add each message with role-based styling
    if 'messages' in _trace:
        for msg in _trace['messages']:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')

            # Create a styled container for each message
            if role == 'user':
                bg_color = '#e3f2fd'
                role_color = '#1976d2'
            elif role == 'assistant':
                bg_color = '#f3e5f5'
                role_color = '#7b1fa2'
            else:
                bg_color = '#f5f5f5'
                role_color = '#616161'

            message_elements.append(
                mo.Html(f"""
                <div style="margin: 2px 0; padding: 4px; background-color: {bg_color}; border-left: 4px solid {role_color};">
                    <div style="font-weight: bold; color: {role_color}; margin-bottom: 2px; text-transform: capitalize;">
                        {role}
                    </div>
                    <div style="white-space: pre-wrap; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
                        {content}
                    </div>
                </div>
                """)
            )

    # Combine all elements
    mo.vstack(message_elements)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 2. Build Transition Matrix""")
    return


@app.cell
def _(labeled_traces):
    transition_tuples = [(t['last_success_state'], t['first_failure_state']) for t in labeled_traces]
    return (transition_tuples,)


@app.cell
def _(pd, transition_tuples):
    # Create transition matrix
    transition_matrix = pd.DataFrame(index=sorted(set(t[0] for t in transition_tuples)), 
                                     columns=sorted(set(t[1] for t in transition_tuples)), 
                                     data=0)
    transition_matrix
    return (transition_matrix,)


@app.cell
def _(Counter, transition_tuples):
    counter = Counter(transition_tuples)
    counter
    return (counter,)


@app.cell
def _(counter, transition_matrix):
    for (last_state, first_failure), count in counter.items():
        transition_matrix.loc[last_state, first_failure] = count
    transition_matrix
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 3. Visualize Transition Matrix""")
    return


@app.cell(hide_code=True)
def _(transition_matrix):
    # Create a heatmap of the transition matrix
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10, 8))
    sns.heatmap(transition_matrix, 
                annot=True, 
                fmt='g', 
                cmap='YlOrRd',
                cbar_kws={'label': 'Count'},
                square=True)
    plt.title('State Transition Matrix Heatmap')
    plt.xlabel('First Failure State')
    plt.ylabel('Last Success State')
    plt.tight_layout()
    plt.gca()
    return plt, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 4. Analyze Transition Matrix""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **High level**

    - GetRecipes is the #1 failure point (35 failures) - This data retrieval service is clearly unstable and needs immediate attention
    - PlanToolCalls succeeds but cascades into lots of failures.  It may be generating faulty plan which causes future tool calls to fail.  Could also just be almost always is called so naturally more failures would go there.
    - GenRecipeArgs and GetRecipes both being very high might mean recipe storage or something very broken about recipe architecture or format

    **Recipe Format or Architecture issue?:**

    - PlanToolCalls → GenRecipeArgs (10 failures) - Planning to recipe argument generation is highly unreliable
    - GetCustomerProfile → GetRecipes (9 failures) - Customer data retrieval failing to access recipes suggests integration issues
    - GenCustomerArgs → GetRecipes (8 failures) - Another recipe access failure pattern

    **Getting Data**

    - `Get` stuff is 40+ failures - Infrastructure/availability issues likely
    - External APIs seems fine since GetWebInfo doesn't fail much, further making me thing internal infra.
    - GetRecipes failures are come from lots of success states - which indicates services issue

    **Generation**

    - Argument Generation services account for a lot of failures.
        - Input validation issue?
        - Does prompt not explain proper format or args well?
    """
    )
    return


@app.cell
def _(Counter, transition_tuples):
    Counter([o[0] for o in transition_tuples]),Counter([o[1] for o in transition_tuples])
    return


@app.cell
def _(Counter, labeled_traces, pd, plt, sns):
    transition_tuples_start = [(t['last_success_state'][:3], t['first_failure_state'][:3]) for t in labeled_traces]

    transition_matrix_start = pd.DataFrame(index=sorted(set(t[0] for t in transition_tuples_start)), 
                                     columns=sorted(set(t[1] for t in transition_tuples_start)), 
                                     data=0)


    counter_start = Counter(transition_tuples_start)
    for (last_state_start, first_failure_start), count_start in counter_start.items():
        transition_matrix_start.loc[last_state_start, first_failure_start] = count_start


    plt.figure(figsize=(10, 8))
    sns.heatmap(transition_matrix_start, 
                annot=True, 
                fmt='g', 
                cmap='YlOrRd',
                cbar_kws={'label': 'Count'},
                square=True)
    plt.title('State Transition Matrix Heatmap')
    plt.xlabel('First Failure State')
    plt.ylabel('Last Success State')
    plt.tight_layout()
    plt.gca()

    return


if __name__ == "__main__":
    app.run()
