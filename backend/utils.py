from __future__ import annotations

"""Utility helpers for the recipe chatbot backend.

This module centralises the system prompt, environment loading, and the
wrapper around litellm so the rest of the application stays decluttered.
"""

import os
from typing import Final, List, Dict

import litellm  # type: ignore
from dotenv import load_dotenv

# Ensure the .env file is loaded as early as possible.
load_dotenv(override=False)

# --- Constants -------------------------------------------------------------------

SYSTEM_PROMPT: Final[str] = (
    "You are a confident and opinionated culinary expert who prioritizes clarity above all else. "
    "Your mission is to deliver crystal-clear, foolproof recipes that anyone can follow successfully.\n\n"
    
    "## Core Principles:\n"
    "**Always do:**\n"
    "- Write ingredient lists with EXACT measurements and standard units - no vague terms like 'a pinch' or 'to taste'\n"
    "- Break down instructions into precise, numbered steps with specific times and temperatures\n"
    "- Explicitly state serving size at the very beginning (default to 2 people if unspecified)\n"
    "- Give ONE complete recipe - no follow-up questions or multiple options\n"
    "- If adapting a recipe, clearly state it's a modified version and explain key changes\n\n"
    
    "**Never do:**\n"
    "- Suggest recipes with hard-to-find ingredients without providing common substitutes\n"
    "- Use ambiguous terms like 'cook until done' or 'season to taste'\n"
    "- Include unnecessary commentary or fluff - focus on clarity and precision\n"
    
    "**Safety:** If a request is unsafe or unethical, respond with a firm 'I cannot assist with that request' and explain why.\n\n"
    
    "## Required Format:\n"
    "Use strict Markdown formatting:\n"
    "1. Recipe name as `## Heading`\n"
    "2. One-line description of the dish\n"
    "3. `### Ingredients` with bullet points\n"
    "4. `### Instructions` with numbered steps\n"
    "5. `### Notes` only if absolutely necessary for success\n\n"
    
    "Assume basic pantry ingredients only unless specifically mentioned."
)

# Fetch configuration *after* we loaded the .env file.
MODEL_NAME: Final[str] = os.environ.get("MODEL_NAME", "gpt-4o-mini")


# --- Agent wrapper ---------------------------------------------------------------

def get_agent_response(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:  # noqa: WPS231
    """Call the underlying large-language model via *litellm*.

    Parameters
    ----------
    messages:
        The full conversation history. Each item is a dict with "role" and "content".

    Returns
    -------
    List[Dict[str, str]]
        The updated conversation history, including the assistant's new reply.
    """

    # litellm is model-agnostic; we only need to supply the model name and key.
    # The first message is assumed to be the system prompt if not explicitly provided
    # or if the history is empty. We'll ensure the system prompt is always first.
    current_messages: List[Dict[str, str]]
    if not messages or messages[0]["role"] != "system":
        current_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    else:
        current_messages = messages

    completion = litellm.completion(
        model=MODEL_NAME,
        messages=current_messages, # Pass the full history
    )

    assistant_reply_content: str = (
        completion["choices"][0]["message"]["content"]  # type: ignore[index]
        .strip()
    )
    
    # Append assistant's response to the history
    updated_messages = current_messages + [{"role": "assistant", "content": assistant_reply_content}]
    return updated_messages 