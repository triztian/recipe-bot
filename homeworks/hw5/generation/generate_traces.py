#!/usr/bin/env python3
"""Synthetic Trace Generator for HW5

This script generates synthetic conversation traces for the Recipe-Chatbot
homework 5.  Each trace intentionally fails at a randomly-sampled pipeline state.

Outputs (written to ../../data/):
• raw_traces.json        – list[dict]  {conversation_id, messages}
• labeled_traces.json    – same objects + last_success_state + first_failure_state

The generation procedure uses **GPT-4.1** twice per trace:
1.  Select a plausible `last_success_state` that precedes a sampled
    `first_failure_state`.
2.  Produce a short conversation (≤ 10 messages) that progresses through the
    pipeline, succeeds through `last_success_state`, and then clearly fails at
    `first_failure_state`.

Environment:
• Requires OPENAI_API_KEY to be set (dotenv supported).
• Install dependencies in homeworks/hw5/requirements.txt (openai, python-dotenv, tqdm).
"""

from __future__ import annotations

import json
import os
import random
import uuid
from pathlib import Path
from typing import Dict, List, Tuple

import litellm
from dotenv import load_dotenv
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# -------------------------------------------------------------
# Configuration
# -------------------------------------------------------------

# Canonical pipeline states (10 total)
PIPELINE_STATES: List[str] = [
    "ParseRequest",
    "PlanToolCalls",
    "GenCustomerArgs",
    "GetCustomerProfile",
    "GenRecipeArgs",
    "GetRecipes",
    "GenWebArgs",
    "GetWebInfo",
    "ComposeResponse",
    "DeliverResponse",
]
STATE_INDEX = {s: i for i, s in enumerate(PIPELINE_STATES)}

# Non-uniform sampling weights for FIRST failure state (must align with list)
FAILURE_WEIGHTS: List[int] = [
    6,  # ParseRequest
    5,  # PlanToolCalls
    10,  # GenCustomerArgs
    12,  # GetCustomerProfile
    15,  # GenRecipeArgs
    30,  # GetRecipes
    7,  # GenWebArgs
    9,  # GetWebInfo
    5,  # ComposeResponse
    1,  # DeliverResponse
]

assert len(FAILURE_WEIGHTS) == len(
    PIPELINE_STATES
), "Weights length must match number of states"

N_TRACES_DEFAULT = 100
MODEL = "gpt-4.1"

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_TRACES_PATH = DATA_DIR / "raw_traces.json"
LABELED_TRACES_PATH = DATA_DIR / "labeled_traces.json"

# -------------------------------------------------------------
# LLM helper via litellm
# -------------------------------------------------------------


def chat_completion(
    messages: List[Dict[str, str]],
    *,
    max_tokens: int = 256,
    temperature: float = 0.7,
    **kwargs,
) -> str:
    """Wrapper around litellm.completion returning content string."""
    resp = litellm.completion(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        **kwargs,
    )
    return resp.choices[0].message.content.strip()


# -------------------------------------------------------------
# Generation steps
# -------------------------------------------------------------

# 1. State sampling helpers ----------------------------------------------------


def pick_first_failure_state() -> str:
    """Sample first_failure_state using predefined weights."""
    return random.choices(PIPELINE_STATES, weights=FAILURE_WEIGHTS, k=1)[0]


def select_last_success_state(first_failure_state: str) -> str:
    """Pick a plausible last_success_state earlier than failure (random predecessor)."""
    idx = STATE_INDEX[first_failure_state]
    if idx == 0:
        return PIPELINE_STATES[0]
    return random.choice(PIPELINE_STATES[:idx])


# 2. Message templates ---------------------------------------------------------

SUCCESS_TEMPLATES: Dict[str, str] = {
    "ParseRequest": "Let me understand your request and preferences…",
    "PlanToolCalls": "I'll gather some additional information before suggesting recipes.",
    "GenCustomerArgs": "TOOL_CALL[GenCustomerArgs] Generating customer profile parameters.",
    "GetCustomerProfile": "TOOL_CALL[GetCustomerProfile] Fetching customer profile.",
    "GenRecipeArgs": "TOOL_CALL[GenRecipeArgs] Preparing recipe search criteria.",
    "GetRecipes": "TOOL_CALL[GetRecipes] Searching recipes now.",
    "GenWebArgs": "TOOL_CALL[GenWebArgs] Building web search query.",
    "GetWebInfo": "TOOL_CALL[GetWebInfo] Pulling additional cooking tips from the web.",
    "ComposeResponse": "I've combined everything—drafting your personalized suggestion…",
    "DeliverResponse": "Here is your customized meal plan!",
}

FAILURE_TEMPLATES: Dict[str, str] = {
    "GenCustomerArgs": "TOOL_CALL[GenCustomerArgs] Error: malformed customer ID parameter.",
    "GetCustomerProfile": "TOOL_CALL[GetCustomerProfile] Error: database timeout (30 s).",
    "GenRecipeArgs": "TOOL_CALL[GenRecipeArgs] Error: token limit exceeded while building query.",
    "GetRecipes": "TOOL_CALL[GetRecipes] Error: no recipes found for given criteria.",
    "GenWebArgs": "TOOL_CALL[GenWebArgs] Error: failed to construct valid search terms.",
    "GetWebInfo": "TOOL_CALL[GetWebInfo] Error: HTTP 503 – service unavailable.",
    "ComposeResponse": "Traceback (most recent call last): KeyError: 'proteinCount' during response assembly.",
    "DeliverResponse": "…",  # DeliverResponse failure will manifest as an empty / partial response.
}


def build_conversation(last_success: str, first_failure: str) -> List[Dict[str, str]]:
    """Construct synthetic conversation messages programmatically."""

    messages: List[Dict[str, str]] = []

    # Opening user message
    opening_user = random.choice(
        [
            "I need a gluten-free dinner idea for four.",
            "Suggest a healthy breakfast using oatmeal.",
            "What vegetarian high-protein meal can I cook tonight?",
        ]
    )
    messages.append({"role": "user", "content": opening_user})

    # Determine state segments
    idx_success = STATE_INDEX[last_success]
    idx_failure = STATE_INDEX[first_failure]

    # States leading to success (inclusive)
    for state in PIPELINE_STATES[: idx_success + 1]:
        messages.append({"role": "agent", "content": SUCCESS_TEMPLATES[state]})

    # Optional mid-conversation user follow-up
    if random.random() < 0.5:
        messages.append({"role": "user", "content": "Sounds good—please continue."})

    # Failure message
    failure_msg = FAILURE_TEMPLATES.get(
        first_failure,
        f"TOOL_CALL[{first_failure}] Error: unexpected failure.",
    )
    messages.append({"role": "agent", "content": failure_msg})

    # Continue with remaining states (success variants) until we hit 8–10 messages
    for state in PIPELINE_STATES[idx_failure + 1 :]:
        if len(messages) >= 9:  # leave room for optional closing user msg
            break
        messages.append({"role": "agent", "content": SUCCESS_TEMPLATES[state]})

    # Closing user acknowledgement to reach 8–10 messages if needed
    if len(messages) < 8:
        messages.append({"role": "user", "content": "Thanks for the help!"})

    # Final length guard
    if not 8 <= len(messages) <= 10:
        raise ValueError("Template conversation length out of bounds")

    return messages


# 3. LLM-based conversation generation ---------------------------------------


def generate_conversation_llm(
    last_success: str, first_failure: str
) -> List[Dict[str, str]]:
    """Use GPT via litellm to craft a coherent conversation trace."""

    # Provide examples for tool call formatting and error examples
    tool_examples = (
        "TOOL_CALL[GetRecipes] Searching recipes now.\n"
        "TOOL_CALL[GetRecipes] Error: no recipes found for given criteria."
    )

    system_prompt = (
        "You are generating *synthetic* conversation traces for a cooking assistant "
        "agent. The agent internally uses the following pipeline states (order is flexible):\n"
        + ", ".join(PIPELINE_STATES)
        + "\n\n"
        "Definitions of special formatting:\n"
        "• When the agent executes a tool, the message MUST start with 'TOOL_CALL[<State>]' as in:\n"
        f"  {tool_examples}\n\n"
        "TASK:\n"
        f"Produce a JSON object with one key 'messages'. The value must be an array of 8–10 messages.\n"
        "Conversation rules:\n"
        "1. First message is from the user.\n"
        "2. For EVERY pipeline state up to and including `{last_success}`, add ONE agent message that represents that state.\n"
        "3. At `{first_failure}` produce an agent message with a realistic error; do NOT apologise.\n"
        "4. After the failure, include 1–2 more agent messages to finish the conversation as if unaware of the failure.\n"
        "5. It is okay to have consecutive agent messages; user may respond once more near the end.\n"
        "6. Do NOT mention state names explicitly except inside TOOL_CALL.\n"
        "Return strictly valid JSON (no markdown)."
    )

    opening_user = random.choice(
        [
            "I need a gluten-free dinner idea for four.",
            "Suggest a healthy breakfast using oatmeal.",
            "What vegetarian high-protein meal can I cook tonight?",
        ]
    )

    messages_input = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"FIRST_FAILURE={first_failure}\nLAST_SUCCESS={last_success}\n"
                f"Start the conversation with this user message: {opening_user}"
            ),
        },
    ]

    response = chat_completion(
        messages_input,
        temperature=0.6,
        max_tokens=500,
        response_format={"type": "json_object"},
    )

    try:
        parsed = json.loads(response)
        msgs = parsed.get("messages", [])
    except Exception as exc:
        raise ValueError(f"LLM returned invalid JSON: {exc}")

    if not 8 <= len(msgs) <= 10:
        raise ValueError("LLM conversation length constraint violated")

    return msgs


# -------------------------------------------------------------
# Main generation routine
# -------------------------------------------------------------


def generate_traces(
    n_traces: int = N_TRACES_DEFAULT,
    seed: int | None = None,
    max_workers: int = 32,
) -> Tuple[List[Dict], List[Dict]]:
    """Generate traces in parallel – returns (raw_traces, labeled_traces)."""

    if seed is not None:
        random.seed(seed)

    raw_traces: List[Dict] = []
    labeled_traces: List[Dict] = []

    def make_trace(_: int, retries: int = 3) -> Tuple[Dict, Dict]:
        """Create one trace (raw, labeled)."""
        attempt = 0
        while True:
            attempt += 1
            first_failure = pick_first_failure_state()
            last_success = select_last_success_state(first_failure)
            try:
                messages = generate_conversation_llm(last_success, first_failure)
                break
            except Exception as exc:
                if attempt >= retries:
                    raise
                continue
        trace_id = str(uuid.uuid4())

        raw_obj = {"conversation_id": trace_id, "messages": messages}
        labeled_obj = {
            **raw_obj,
            "last_success_state": last_success,
            "first_failure_state": first_failure,
        }
        return raw_obj, labeled_obj

    workers = min(max_workers, n_traces)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(make_trace, i) for i in range(n_traces)]
        for fut in tqdm(
            as_completed(futures), total=n_traces, desc="Generating traces"
        ):
            try:
                raw_obj, labeled_obj = fut.result()
                raw_traces.append(raw_obj)
                labeled_traces.append(labeled_obj)
            except Exception as exc:
                print(f"[WARN] trace generation failed: {exc}")

    return raw_traces, labeled_traces


# -------------------------------------------------------------
# CLI Entry
# -------------------------------------------------------------


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic HW5 traces.")
    parser.add_argument(
        "--n",
        type=int,
        default=N_TRACES_DEFAULT,
        help="number of traces to generate (default 100)",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="random seed for reproducibility"
    )
    args = parser.parse_args()

    # Load env vars
    load_dotenv()
    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("OPENAI_API_KEY not set in environment")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    raw, labeled = generate_traces(args.n, args.seed)

    with open(RAW_TRACES_PATH, "w") as f_raw:
        json.dump(raw, f_raw, indent=2)
    with open(LABELED_TRACES_PATH, "w") as f_lab:
        json.dump(labeled, f_lab, indent=2)

    print(f"\nWrote {len(raw)} traces to {RAW_TRACES_PATH.relative_to(Path.cwd())}")
    print(f"Wrote labeled traces to {LABELED_TRACES_PATH.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
