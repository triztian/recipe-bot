#!/usr/bin/env python3
"""
Convert synthetic traces from JSON to CSV format for inspection interface.

This script reads the synthetic_traces.json file from hw5 and converts it to a CSV
with columns for trace_id, customer_persona, failure_mode, user_query, and formatted messages.
"""

import json
import csv
import os
from pathlib import Path
from typing import List, Dict, Any


def format_messages(messages: List[Dict[str, Any]]) -> str:
    """
    Format conversation messages into a readable string.
    
    Args:
        messages: List of message dictionaries from the trace
        
    Returns:
        Formatted string with all messages concatenated
    """
    formatted_parts = []
    
    for msg in messages:
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')
        tool_name = msg.get('tool_name')
        
        if role == 'user':
            formatted_parts.append(f"USER: {content}")
        elif role == 'agent':
            formatted_parts.append(f"AGENT: {content}")
        elif role == 'tool':
            tool_info = f" ({tool_name})" if tool_name else ""
            formatted_parts.append(f"TOOL{tool_info}: {content}")
        else:
            formatted_parts.append(f"{role.upper()}: {content}")
    
    return " | ".join(formatted_parts)


def extract_user_query(messages: List[Dict[str, Any]]) -> str:
    """
    Extract the initial user query from the conversation.
    
    Args:
        messages: List of message dictionaries from the trace
        
    Returns:
        The first user message content, or empty string if not found
    """
    for msg in messages:
        if msg.get('role') == 'user':
            return msg.get('content', '')
    return ''


def extract_tool_calls(messages: List[Dict[str, Any]]) -> str:
    """
    Extract and summarize tool calls from the conversation.
    
    Args:
        messages: List of message dictionaries from the trace
        
    Returns:
        Summary of tool calls made during the conversation
    """
    tool_calls = []
    
    for msg in messages:
        if msg.get('role') == 'tool' and msg.get('tool_name'):
            tool_name = msg.get('tool_name')
            tool_input = msg.get('tool_input', {})
            tool_output = msg.get('tool_output', {})
            
            # Create a concise summary
            input_summary = str(tool_input)[:100] + "..." if len(str(tool_input)) > 100 else str(tool_input)
            output_summary = str(tool_output)[:100] + "..." if len(str(tool_output)) > 100 else str(tool_output)
            
            tool_calls.append(f"{tool_name}(input: {input_summary}, output: {output_summary})")
    
    return " | ".join(tool_calls)


def convert_traces_to_csv(input_file: str, output_file: str) -> None:
    """
    Convert synthetic traces JSON to CSV format.
    
    Args:
        input_file: Path to the synthetic_traces.json file
        output_file: Path to write the output CSV file
    """
    print(f"Reading traces from: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    traces = data.get('traces', [])
    print(f"Found {len(traces)} traces to convert")
    
    # Define CSV columns
    fieldnames = [
        'trace_id',
        'customer_persona', 
        'user_query',
        'conversation_messages',
        'tool_calls'
    ]
    
    print(f"Writing CSV to: {output_file}")
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for trace in traces:
            messages = trace.get('messages', [])
            
            row = {
                'trace_id': trace.get('trace_id', ''),
                'customer_persona': trace.get('customer_persona', ''),
                'user_query': extract_user_query(messages),
                'conversation_messages': format_messages(messages),
                'tool_calls': extract_tool_calls(messages)
            }
            
            writer.writerow(row)
    
    print(f"Successfully converted {len(traces)} traces to CSV")


def main():
    """Main function to run the conversion."""
    # Set up paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    # Input file from hw5
    input_file = project_root / "homeworks" / "hw5" / "data" / "synthetic_traces.json"
    
    # Output file in lesson-7
    output_file = script_dir.parent / "data" / "traces.csv"
    
    # Check if input file exists
    if not input_file.exists():
        print(f"Error: Input file not found at {input_file}")
        print("Make sure you're running this from the correct directory and hw5 data exists.")
        return
    
    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert traces
    convert_traces_to_csv(str(input_file), str(output_file))
    
    print(f"\nConversion complete! CSV file created at: {output_file}")
    print("\nYou can now inspect the traces using any CSV viewer or the upcoming inspection interface.")


if __name__ == "__main__":
    main() 