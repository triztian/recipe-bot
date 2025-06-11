# Lesson 6: Custom Failure Mode Inspection Interface

## Overview

This lesson builds a custom interface to inspect and manually mark failure modes in the synthetic conversation traces from Homework 5. The goal is to create tools for detailed analysis of where and how conversational agents fail.

## Directory Structure

```
lesson-7/
├── data/
│   ├── traces.csv                    # Converted CSV from synthetic traces
│   └── inspection_results.json      # Results from manual inspection (future)
├── scripts/
│   ├── convert_traces_to_csv.py     # Convert JSON traces to CSV
│   └── inspection_interface.py      # Interactive interface (future)
├── requirements.txt                 # Python dependencies
└── README.md                       # This file
```

## Getting Started

### 1. Install Dependencies

```bash
cd lesson-7
pip install -r requirements.txt
```

### 2. Convert Traces to CSV

The first step is to convert the synthetic traces from HW5 into a more manageable CSV format:

```bash
python scripts/convert_traces_to_csv.py
```

This script will:
- Read `homeworks/hw5/data/synthetic_traces.json`
- Convert it to a CSV with the following columns:
  - `trace_id`: Unique identifier for each conversation
  - `customer_persona`: Customer type (e.g., "gluten_free_family")
  - `failure_mode`: Identified failure type
  - `failure_category`: High-level failure category
  - `user_query`: The initial user message
  - `conversation_messages`: All messages formatted as "ROLE: content"
  - `tool_calls`: Summary of tool calls made
  - `overall_success`: Whether the conversation succeeded
- Save the result to `data/traces.csv`

### 3. Inspect the CSV

Once converted, you can open `data/traces.csv` in any spreadsheet application or CSV viewer to inspect the conversations and their failure modes.

## CSV Format Details

The conversion script formats the data for easy human inspection:

- **Messages are concatenated** with role prefixes (USER:, AGENT:, TOOL:)
- **Tool calls are summarized** with input/output information
- **User queries are extracted** from the first user message
- **All metadata is preserved** from the original traces

## Future Features

- Interactive web interface for marking additional failure modes
- Bulk annotation tools
- Export annotated results for further analysis
- Integration with the transition heatmap analysis from HW5

## Usage Example

```bash
# Convert traces
python scripts/convert_traces_to_csv.py

# Output will be in data/traces.csv
# Open with Excel, Google Sheets, or any CSV viewer for inspection
```

The resulting CSV will have 118 rows (one for each failed conversation trace) with easy-to-read conversation data for manual inspection and analysis. 