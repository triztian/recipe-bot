#!/usr/bin/env python3
"""
Transition Heatmap Analysis Utilities

Analyzes conversation traces to generate transition matrices and heatmaps
showing how conversations flow between agent pipeline states.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple
from pathlib import Path
from collections import defaultdict, Counter
import sys
import litellm
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Add hw5 root to path for consistent imports
SCRIPT_DIR = Path(__file__).parent
HW5_ROOT = SCRIPT_DIR.parent
sys.path.append(str(HW5_ROOT))

# Load environment variables
load_dotenv()

class TransitionAnalyzer:
    """Analyzes conversation traces for agent pipeline state transitions."""
    
    def __init__(self):
        self.traces = []
        self.transition_matrix = None
        
        # Define tool-specific agent pipeline states
        self.agent_states = [
            "FetchCustomer",           # Getting customer data/preferences
            "CustomerToolError",       # Customer DB tool failures
            "ParseCustomerResults",    # Processing customer tool results
            
            "FetchRecipes",            # Searching for recipes
            "RecipeToolError",         # Recipe search tool failures  
            "ParseRecipeResults",      # Processing recipe search results
            
            "FetchInternet",           # Searching internet for info
            "InternetToolError",       # Internet search tool failures
            "ParseInternetResults",    # Processing internet search results
            
            "FetchDietary",            # Getting dietary restrictions
            "DietaryToolError",        # Dietary tool failures
            "ParseDietaryResults",     # Processing dietary tool results
            
            "GenerateFinalResponse"    # Creating final response to user
        ]
        
        self.state_mapping = {state: i for i, state in enumerate(self.agent_states)}
        
    def load_traces(self, traces_file: str):
        """Load conversation traces from JSON file."""
        with open(traces_file, 'r') as f:
            data = json.load(f)
            self.traces = data.get("traces", [])
        print(f"Loaded {len(self.traces)} traces for pipeline analysis")
    
    def classify_trace_states_with_llm(self, trace: Dict[str, Any]) -> List[str]:
        """Use LLM to classify the actual pipeline states for a conversation trace."""
        
        # Create a summary of the conversation for the LLM
        conversation_summary = []
        for msg in trace["messages"]:
            role = msg["role"]
            content = msg["content"]
            
            if role == "tool":
                tool_name = msg.get("tool_name", "unknown")
                tool_error = msg.get("tool_output", {}).get("error")
                if tool_error:
                    conversation_summary.append(f"TOOL({tool_name}): ERROR - {tool_error}")
                else:
                    conversation_summary.append(f"TOOL({tool_name}): {content}")
            else:
                failure_indicators = msg.get("failure_indicators", [])
                if failure_indicators:
                    conversation_summary.append(f"{role.upper()}: {content} [FAILURE: {failure_indicators}]")
                else:
                    conversation_summary.append(f"{role.upper()}: {content}")
        
        conversation_text = "\n".join(conversation_summary)
        
        prompt = f"""
Analyze this cooking assistant conversation and identify WHERE THE FIRST FAILURE occurred in the agent pipeline.

CONVERSATION:
{conversation_text}

AVAILABLE STATES:
- FetchCustomer: Getting customer data/preferences from customer database
- CustomerToolError: Customer database tool failures
- ParseCustomerResults: Processing customer database results

- FetchRecipes: Searching for recipes using recipe database
- RecipeToolError: Recipe search tool failures  
- ParseRecipeResults: Processing recipe search results

- FetchInternet: Searching internet for cooking information
- InternetToolError: Internet search tool failures
- ParseInternetResults: Processing internet search results

- FetchDietary: Getting dietary restrictions for customer
- DietaryToolError: Dietary restriction tool failures
- ParseDietaryResults: Processing dietary restriction results

- GenerateFinalResponse: Creating final response to user

Look at the conversation and identify:
1. What specific tool operation was being performed when the first failure occurred?
2. What state did the failure transition to?

Return ONLY two states separated by a comma: FromState,ToState

Examples:
- If customer DB call failed: FetchCustomer,CustomerToolError
- If recipe search failed: FetchRecipes,RecipeToolError
- If processing recipe results failed: ParseRecipeResults,RecipeToolError
- If internet search failed: FetchInternet,InternetToolError
- If final response generation failed: GenerateFinalResponse,RecipeToolError

Focus on the FIRST failure that occurred and which specific tool was involved.

States:"""

        try:
            response = litellm.completion(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=100
            )
            
            states_text = response.choices[0].message.content.strip()
            
            # Parse the response - expecting "FromState,ToState"
            if "," in states_text:
                states = [s.strip() for s in states_text.split(",")]
                if len(states) == 2:
                    # Validate both states
                    valid_states = [s for s in states if s in self.agent_states]
                    if len(valid_states) == 2:
                        return valid_states
            
            # If parsing failed, try fallback
            return self._fallback_state_detection(trace)
            
        except Exception as e:
            print(f"Error classifying trace states: {e}")
            return self._fallback_state_detection(trace)
    
    def _fallback_state_detection(self, trace: Dict[str, Any]) -> List[str]:
        """Simple fallback state detection if LLM fails - returns first failure transition."""
        messages = trace["messages"]
        
        # Look for tool errors by tool name
        for msg in messages:
            if msg["role"] == "tool" and (msg.get("tool_output", {}).get("error") or msg.get("failure_indicators")):
                tool_name = msg.get("tool_name", "")
                
                if "customer" in tool_name.lower():
                    return ["FetchCustomer", "CustomerToolError"]
                elif "recipe" in tool_name.lower():
                    return ["FetchRecipes", "RecipeToolError"] 
                elif "internet" in tool_name.lower():
                    return ["FetchInternet", "InternetToolError"]
                elif "dietary" in tool_name.lower():
                    return ["FetchDietary", "DietaryToolError"]
                else:
                    return ["FetchRecipes", "RecipeToolError"]  # Default to recipe
        
        # Look for agent failure indicators
        for msg in messages:
            if msg.get("failure_indicators"):
                if msg["role"] == "agent":
                    # Try to guess what tool was being used based on content
                    content = msg["content"].lower()
                    if any(word in content for word in ["customer", "profile", "preference"]):
                        return ["ParseCustomerResults", "CustomerToolError"]
                    elif any(word in content for word in ["recipe", "cooking", "ingredient"]):
                        return ["ParseRecipeResults", "RecipeToolError"]
                    elif any(word in content for word in ["search", "internet", "online"]):
                        return ["ParseInternetResults", "InternetToolError"]
                    elif any(word in content for word in ["dietary", "allerg", "restriction"]):
                        return ["ParseDietaryResults", "DietaryToolError"]
                    else:
                        return ["GenerateFinalResponse", "RecipeToolError"]
        
        # Default fallback
        return ["FetchRecipes", "RecipeToolError"]
    
    def build_failure_transition_matrix(self) -> np.ndarray:
        """Build matrix showing where failures occur in state transitions."""
        n_states = len(self.agent_states)
        failure_counts = np.zeros((n_states, n_states), dtype=int)
        
        # Get only failed traces
        failed_traces = [trace for trace in self.traces if not trace["overall_success"]]
        
        if not failed_traces:
            print("No failed traces found!")
            return failure_counts
        
        print(f"Analyzing {len(failed_traces)} failed traces with parallel LLM state classification...")
        
        # Process traces in parallel
        def process_single_trace(trace):
            """Process a single trace and return its first failure transition."""
            try:
                states = self.classify_trace_states_with_llm(trace)
                if len(states) != 2:
                    return None
                
                # Return the single failure transition
                from_state = self.state_mapping[states[0]]
                to_state = self.state_mapping[states[1]]
                return (from_state, to_state)
                
            except Exception as e:
                print(f"Error processing trace {trace.get('trace_id', 'unknown')}: {e}")
                return None
        
        # Use ThreadPoolExecutor for parallel processing
        max_workers = min(16, len(failed_traces))  # Limit concurrent requests
        all_transitions = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_trace = {
                executor.submit(process_single_trace, trace): trace
                for trace in failed_traces
            }
            
            # Collect results with progress bar
            for future in tqdm(as_completed(future_to_trace), 
                             total=len(failed_traces), 
                             desc="Processing traces"):
                try:
                    transition = future.result()
                    if transition:
                        all_transitions.append(transition)
                except Exception as e:
                    print(f"Failed to process trace: {e}")
        
        # Count all transitions
        for from_state, to_state in all_transitions:
            failure_counts[from_state, to_state] += 1
        
        print(f"Processed {len(all_transitions)} failure transitions")
        return failure_counts
    
    def create_failure_transition_heatmap(self, failure_matrix: np.ndarray, output_path: str):
        """Create the main failure transition heatmap."""
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(
            failure_matrix,
            annot=True,
            fmt='d',
            cmap='Reds',
            xticklabels=self.agent_states,
            yticklabels=self.agent_states,
            cbar_kws={'label': 'Failure Count'},
            square=True
        )
        
        plt.title('Failure Occurred In State →', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('To State', fontsize=12)
        plt.ylabel('From State ↓', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save plot
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved failure transition heatmap: {output_path}")
        
        # Print summary
        total_failures = np.sum(failure_matrix)
        if total_failures > 0:
            print(f"\nFailure Analysis Summary:")
            print(f"Total failure transitions: {total_failures}")
            
            # Find most common failure transitions
            max_failures = np.max(failure_matrix)
            if max_failures > 0:
                max_indices = np.where(failure_matrix == max_failures)
                for i, j in zip(max_indices[0], max_indices[1]):
                    from_state = self.agent_states[i]
                    to_state = self.agent_states[j]
                    print(f"Most common failure transition: {from_state} → {to_state} ({max_failures} failures)")

    def analyze_failure_traces(self, output_dir: str = None):
        """Main analysis function - generates the failure transition heatmap."""
        print("Starting failure transition analysis...")
        
        # Default output directory
        if output_dir is None:
            output_dir = str(HW5_ROOT / "results" / "visualizations")
        
        # Build failure transition matrix
        failure_matrix = self.build_failure_transition_matrix()
        
        # Create the heatmap
        self.create_failure_transition_heatmap(
            failure_matrix,
            f"{output_dir}/failure_transition_heatmap.png"
        )
        
        # Save analysis data
        results = {
            "failure_transition_matrix": failure_matrix.tolist(),
            "agent_states": self.agent_states,
            "total_traces_analyzed": len(self.traces),
            "failed_traces": len([t for t in self.traces if not t["overall_success"]])
        }
        
        results_file = f"{Path(output_dir).parent}/transition_analysis_results.json"
        Path(results_file).parent.mkdir(parents=True, exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Analysis complete! Results saved to {results_file}")
        return results

def main():
    """Main function for running transition analysis."""
    # Path resolution
    DATA_DIR = HW5_ROOT / "data"
    
    # Check if traces file exists
    traces_file = DATA_DIR / "synthetic_traces.json"
    if not traces_file.exists():
        print(f"Error: {traces_file} not found!")
        print("Please run generate_failure_traces.py first")
        return
    
    # Initialize analyzer
    analyzer = TransitionAnalyzer()
    analyzer.load_traces(str(traces_file))
    
    # Run analysis
    results = analyzer.analyze_failure_traces()
    
    print(f"\nAnalysis Summary:")
    print(f"- Total traces analyzed: {results['total_traces_analyzed']}")
    print(f"- Failed traces: {results['failed_traces']}")
    print(f"- Agent pipeline states: {len(results['agent_states'])}")

if __name__ == "__main__":
    main() 