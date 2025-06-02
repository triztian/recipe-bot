#!/usr/bin/env python3
"""
Generate Synthetic Failure Traces for Agent Analysis

This script generates realistic conversation traces demonstrating agent failure modes
using a two-phase approach:
1. Generate specific failure scenario descriptions
2. Generate multi-turn conversations demonstrating those failures
"""

import json
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

import litellm
from dotenv import load_dotenv
from tqdm import tqdm
from pydantic import BaseModel, field_validator

# Load environment variables
load_dotenv()

# Configuration
MODEL_NAME = "gpt-4o"
MAX_WORKERS = 32
TRACES_TO_GENERATE = 120

# Path resolution - get hw5 root directory from script location
SCRIPT_DIR = Path(__file__).parent
HW5_ROOT = SCRIPT_DIR.parent
DATA_DIR = HW5_ROOT / "data"
OUTPUT_FILE = DATA_DIR / "synthetic_traces.json"

class TraceMessage(BaseModel):
    """Individual message in a conversation trace."""
    role: str  # "user", "agent", "tool"
    content: str
    tool_name: Optional[str] = None
    tool_input: Optional[Union[Dict, str]] = None
    tool_output: Optional[Union[Dict, str]] = None
    timestamp: str
    failure_indicators: Optional[List[str]] = None
    recovery_attempted: Optional[bool] = None
    
    @field_validator('tool_input', 'tool_output', mode='before')
    @classmethod
    def parse_tool_data(cls, v):
        """Convert string tool data to proper format."""
        if v is None:
            return None
        if isinstance(v, str):
            # Try to parse as JSON first
            try:
                import json
                return json.loads(v)
            except (json.JSONDecodeError, TypeError):
                # If it's just a string, wrap it appropriately
                if v.strip():
                    return {"data": v}
                return None
        return v

class ConversationTrace(BaseModel):
    """Complete conversation trace with metadata."""
    trace_id: str
    failure_mode: str
    customer_persona: str
    messages: List[TraceMessage]
    overall_success: bool
    failure_category: str
    recovery_success: Optional[bool] = None
    generated_at: str

class FailureTraceGenerator:
    """Generates synthetic conversation traces demonstrating agent failures."""
    
    def __init__(self):
        self.failure_modes = self._load_failure_modes()
        self.customer_personas = self._load_customer_personas()
        
    def _load_failure_modes(self) -> Dict[str, Any]:
        """Load failure mode definitions."""
        with open(DATA_DIR / "failure_modes.json", "r") as f:
            return json.load(f)
    
    def _load_customer_personas(self) -> Dict[str, Any]:
        """Load customer persona definitions."""
        with open(DATA_DIR / "customer_personas.json", "r") as f:
            return json.load(f)
    
    def _call_llm(self, messages: List[Dict[str, str]], max_retries: int = 3) -> str:
        """Call LLM with retry logic."""
        for attempt in range(max_retries):
            try:
                response = litellm.completion(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0.8,  # Higher temperature for more diverse failures
                    max_tokens=2000
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(2 ** attempt)  # Exponential backoff
    
    def generate_failure_scenario(self, failure_mode: Dict[str, Any], persona: Dict[str, Any]) -> str:
        """Generate a specific failure scenario description."""
        scenario_prompt = f"""
You are designing a test case for an AI cooking assistant that has access to these tools:
- retrieve_recipes(query, filters): Search recipe database
- query_customer_db(customer_id, query_type): Get customer preferences/history
- search_internet(query): Search web for cooking information  
- get_dietary_restrictions(customer_id): Get customer dietary info

Create a specific failure scenario based on:

FAILURE MODE: {failure_mode['id']}
Description: {failure_mode['description']}
Category: {failure_mode['category']}
Trigger: {failure_mode['trigger']}
Expected Agent Behavior: {failure_mode['agent_behavior']}

CUSTOMER PERSONA: {persona['name']} ({persona['persona_id']})
- Cooking Skill: {persona['cooking_skill']}
- Dietary Restrictions: {persona['dietary_restrictions']}
- Allergies: {persona['allergies']}
- Preferences: {json.dumps(persona['preferences'], indent=2)}
- Communication Style: {json.dumps(persona['conversation_style'], indent=2)}

Write a 2-3 sentence scenario description that explains:
1. What the customer will ask about
2. What specific failure will occur (tool error, agent mistake, etc.)
3. How this failure will manifest in the conversation

Make it realistic and specific to this customer's cooking needs and communication style.
"""

        messages = [{"role": "user", "content": scenario_prompt}]
        return self._call_llm(messages)
    
    def generate_conversation_trace(self, scenario: str, failure_mode: Dict[str, Any], persona: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate a full conversation trace demonstrating the failure."""
        
        # Create more specific prompts based on failure category
        failure_category = failure_mode.get("category", "")
        failure_id = failure_mode.get("id", "")
        
        # Determine which tool should be used and how it should fail
        tool_scenarios = {
            "empty_recipe_hallucination": ("retrieve_recipes", "empty_results", "ParseRecipeResults"),
            "dietary_restriction_ignored": ("get_dietary_restrictions", "success_but_ignored", "ParseDietaryResults"),
            "customer_data_timeout_ignored": ("query_customer_db", "timeout", "FetchCustomer"),
            "internet_search_hallucination": ("search_internet", "empty_results", "ParseInternetResults"),
            "recipe_query_too_broad": ("retrieve_recipes", "irrelevant_results", "ParseRecipeResults"),
            "tool_chain_breakdown": ("query_customer_db", "partial_success", "ParseCustomerResults"),
            "context_loss_in_conversation": ("retrieve_recipes", "success_but_context_lost", "GenerateFinalResponse"),
            "wrong_tool_for_task": ("search_internet", "wrong_tool_used", "FetchInternet"),
            "error_message_not_handled": ("retrieve_recipes", "error_not_handled", "FetchRecipes"),
            "circular_tool_calling": ("retrieve_recipes", "repeated_calls", "FetchRecipes"),
            "malformed_query_parameters": ("query_customer_db", "malformed_params", "FetchCustomer"),
            "preference_contradiction": ("get_dietary_restrictions", "contradictory_data", "ParseDietaryResults")
        }
        
        tool_name, failure_type, expected_state = tool_scenarios.get(failure_id, ("retrieve_recipes", "generic_error", "FetchRecipes"))
        
        trace_prompt = f"""
Generate a realistic cooking assistant conversation that demonstrates this specific failure:

FAILURE SCENARIO: {scenario}
CUSTOMER: {persona['name']} - {persona['conversation_style']['communication']} style, {persona['conversation_style']['patience']} patience
FAILURE TYPE: {failure_mode['description']}

SPECIFIC REQUIREMENTS:
- Primary tool to use: {tool_name}
- Failure should occur during: {expected_state}
- Failure type: {failure_type}

Create a JSON conversation trace that shows:
1. User makes a request
2. Agent decides to use {tool_name}
3. {self._get_failure_instruction(failure_type, tool_name)}
4. Agent exhibits the specific failure behavior: {failure_mode['agent_behavior']}

JSON Format:
[
  {{
    "role": "user",
    "content": "realistic request from {persona['name']}",
    "timestamp": "2024-01-15T10:00:00Z"
  }},
  {{
    "role": "agent",
    "content": "I'll help you with that. Let me check...",
    "timestamp": "2024-01-15T10:00:03Z"
  }},
  {{
    "role": "tool",
    "tool_name": "{tool_name}",
    "tool_input": {{"appropriate": "parameters"}},
    "tool_output": {self._get_tool_output_example(failure_type)},
    "content": "Tool execution result",
    "timestamp": "2024-01-15T10:00:05Z"
  }},
  {{
    "role": "agent",
    "content": "agent response showing the failure behavior",
    "failure_indicators": ["{failure_id}"],
    "timestamp": "2024-01-15T10:00:08Z"
  }}
]

Make the conversation realistic for {persona['name']} and clearly demonstrate {failure_mode['id']}.
"""

        messages = [{"role": "user", "content": trace_prompt}]
        response = self._call_llm(messages)
        
        # Parse JSON response with better error handling
        try:
            # Extract JSON from response (handle markdown code blocks)
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            elif "[" in response and "]" in response:
                json_start = response.find("[")
                json_end = response.rfind("]") + 1
                json_str = response[json_start:json_end]
            else:
                json_str = response
            
            parsed_messages = json.loads(json_str)
            
            # Validate and clean each message
            cleaned_messages = []
            for msg in parsed_messages:
                # Ensure required fields exist
                if not all(key in msg for key in ["role", "content", "timestamp"]):
                    continue
                    
                # Clean tool_input and tool_output
                if "tool_input" in msg and isinstance(msg["tool_input"], str):
                    # Try to convert string to dict format
                    if msg["tool_input"].strip():
                        msg["tool_input"] = {"query": msg["tool_input"]}
                    else:
                        msg["tool_input"] = None
                        
                if "tool_output" in msg and isinstance(msg["tool_output"], str):
                    # Try to convert string to dict format
                    if msg["tool_output"].strip():
                        msg["tool_output"] = {"result": msg["tool_output"]}
                    else:
                        msg["tool_output"] = None
                
                cleaned_messages.append(msg)
                
            return cleaned_messages
            
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {e}")
            print(f"Response was: {response[:500]}...")
            return []
        except Exception as e:
            print(f"Error processing response: {e}")
            return []
    
    def _get_failure_instruction(self, failure_type: str, tool_name: str) -> str:
        """Get specific instructions for how the failure should occur."""
        instructions = {
            "empty_results": f"{tool_name} returns empty results, but agent makes up information",
            "timeout": f"{tool_name} times out, but agent ignores the error",
            "irrelevant_results": f"{tool_name} returns irrelevant results that agent misinterprets",
            "partial_success": f"{tool_name} returns incomplete data, agent doesn't handle properly",
            "success_but_ignored": f"{tool_name} succeeds but agent ignores important information",
            "success_but_context_lost": f"{tool_name} succeeds but agent loses conversation context",
            "wrong_tool_used": f"Agent uses {tool_name} when a different tool would be appropriate",
            "error_not_handled": f"{tool_name} returns error but agent doesn't explain to user",
            "repeated_calls": f"Agent calls {tool_name} multiple times with same parameters",
            "malformed_params": f"Agent sends malformed parameters to {tool_name}",
            "contradictory_data": f"{tool_name} returns data that contradicts earlier information",
            "generic_error": f"{tool_name} fails in a generic way"
        }
        return instructions.get(failure_type, f"{tool_name} fails somehow")
    
    def _get_tool_output_example(self, failure_type: str) -> str:
        """Get example tool output for different failure types."""
        outputs = {
            "empty_results": '{"recipes": [], "message": "No recipes found"}',
            "timeout": '{"error": "Request timeout after 30 seconds"}',
            "irrelevant_results": '{"recipes": [{"name": "Unrelated Recipe", "id": "123"}]}',
            "partial_success": '{"customer_data": {"name": "John"}, "preferences": null}',
            "success_but_ignored": '{"dietary_restrictions": ["gluten-free", "nut-free"]}',
            "success_but_context_lost": '{"recipes": [{"name": "Pasta Recipe", "ingredients": ["pasta"]}]}',
            "wrong_tool_used": '{"search_results": ["Generic cooking info"]}',
            "error_not_handled": '{"error": "Database connection failed"}',
            "repeated_calls": '{"recipes": [], "message": "Still no results"}',
            "malformed_params": '{"error": "Invalid query format"}',
            "contradictory_data": '{"allergies": ["dairy"], "preferences": ["cheese-heavy dishes"]}',
            "generic_error": '{"error": "Something went wrong"}'
        }
        return outputs.get(failure_type, '{"error": "Unknown error"}')
    
    def generate_single_trace(self, failure_mode_id: str, persona_id: str) -> Optional[ConversationTrace]:
        """Generate a single conversation trace."""
        try:
            # Get failure mode and persona
            failure_mode = next(fm for fm in self.failure_modes["specific_failure_modes"] 
                              if fm["id"] == failure_mode_id)
            persona = next(p for p in self.customer_personas["personas"]
                          if p["persona_id"] == persona_id)
            
            # Generate scenario
            scenario = self.generate_failure_scenario(failure_mode, persona)
            
            # Generate conversation
            raw_messages = self.generate_conversation_trace(scenario, failure_mode, persona)
            
            if not raw_messages:
                return None
            
            # Convert to TraceMessage objects with error handling
            messages = []
            for msg in raw_messages:
                try:
                    messages.append(TraceMessage(
                        role=msg["role"],
                        content=msg["content"],
                        tool_name=msg.get("tool_name"),
                        tool_input=msg.get("tool_input"),
                        tool_output=msg.get("tool_output"),
                        timestamp=msg["timestamp"],
                        failure_indicators=msg.get("failure_indicators"),
                        recovery_attempted=msg.get("recovery_attempted")
                    ))
                except Exception as e:
                    print(f"Error creating TraceMessage: {e}")
                    print(f"Problematic message: {msg}")
                    # Skip this message and continue
                    continue
            
            if not messages:
                print(f"No valid messages created for {failure_mode_id} + {persona_id}")
                return None
            
            # Determine overall success and recovery
            overall_success = not any(msg.failure_indicators for msg in messages if msg.failure_indicators)
            recovery_success = any(msg.recovery_attempted for msg in messages if msg.recovery_attempted)
            
            return ConversationTrace(
                trace_id=str(uuid.uuid4()),
                failure_mode=failure_mode_id,
                customer_persona=persona_id,
                messages=messages,
                overall_success=overall_success,
                failure_category=failure_mode["category"],
                recovery_success=recovery_success if recovery_success else None,
                generated_at=datetime.now().isoformat()
            )
            
        except Exception as e:
            print(f"Error generating trace for {failure_mode_id} + {persona_id}: {e}")
            return None
    
    def generate_trace_combinations(self) -> List[tuple]:
        """Generate combinations of failure modes and personas for trace generation."""
        failure_mode_ids = [fm["id"] for fm in self.failure_modes["specific_failure_modes"]]
        persona_ids = [p["persona_id"] for p in self.customer_personas["personas"]]
        
        # Group failure modes by the tool they should use
        tool_groups = {
            "customer_tools": ["dietary_restriction_ignored", "customer_data_timeout_ignored", "tool_chain_breakdown", "malformed_query_parameters"],
            "recipe_tools": ["empty_recipe_hallucination", "recipe_query_too_broad", "error_message_not_handled", "circular_tool_calling"],
            "internet_tools": ["internet_search_hallucination", "wrong_tool_for_task"],
            "dietary_tools": ["preference_contradiction"],
            "multi_step": ["context_loss_in_conversation"]
        }
        
        combinations = []
        
        # Ensure balanced representation of each tool type
        for tool_group, modes in tool_groups.items():
            # Each mode in this group gets 8-12 traces
            for mode in modes:
                if mode in failure_mode_ids:
                    num_traces = random.randint(8, 12)
                    # Distribute across different personas
                    selected_personas = random.choices(persona_ids, k=num_traces)
                    for persona in selected_personas:
                        combinations.append((mode, persona))
        
        # Add some random additional combinations to reach target
        remaining = TRACES_TO_GENERATE - len(combinations)
        if remaining > 0:
            for _ in range(remaining):
                failure_mode_id = random.choice(failure_mode_ids)
                persona_id = random.choice(persona_ids)
                combinations.append((failure_mode_id, persona_id))
        
        # Shuffle to avoid clustering
        random.shuffle(combinations)
        return combinations[:TRACES_TO_GENERATE]
    
    def generate_all_traces(self) -> List[ConversationTrace]:
        """Generate all conversation traces using parallel processing."""
        combinations = self.generate_trace_combinations()
        
        print(f"Generating {len(combinations)} failure traces...")
        print(f"Using {MAX_WORKERS} workers for parallel processing")
        
        traces = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all tasks
            future_to_combo = {
                executor.submit(self.generate_single_trace, failure_mode_id, persona_id): (failure_mode_id, persona_id)
                for failure_mode_id, persona_id in combinations
            }
            
            # Collect results with progress bar
            for future in tqdm(as_completed(future_to_combo), total=len(combinations), desc="Generating traces"):
                combo = future_to_combo[future]
                try:
                    trace = future.result()
                    if trace:
                        traces.append(trace)
                    else:
                        print(f"Failed to generate trace for {combo}")
                except Exception as e:
                    print(f"Exception for {combo}: {e}")
        
        print(f"Successfully generated {len(traces)} traces")
        return traces
    
    def save_traces(self, traces: List[ConversationTrace], output_path: str):
        """Save traces to JSON file."""
        # Convert to dictionaries for JSON serialization
        traces_data = [trace.model_dump() for trace in traces]
        
        # Create metadata
        metadata = {
            "generated_at": datetime.now().isoformat(),
            "total_traces": len(traces),
            "model_used": MODEL_NAME,
            "traces_by_category": {},
            "traces_by_persona": {}
        }
        
        # Calculate statistics
        for trace in traces:
            category = trace.failure_category
            persona = trace.customer_persona
            
            if category not in metadata["traces_by_category"]:
                metadata["traces_by_category"][category] = 0
            metadata["traces_by_category"][category] += 1
            
            if persona not in metadata["traces_by_persona"]:
                metadata["traces_by_persona"][persona] = 0
            metadata["traces_by_persona"][persona] += 1
        
        # Save to file
        output_data = {
            "metadata": metadata,
            "traces": traces_data
        }
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Saved {len(traces)} traces to {output_path}")
        print(f"Traces by category: {metadata['traces_by_category']}")
        print(f"Traces by persona: {metadata['traces_by_persona']}")

def main():
    """Main execution function."""
    print("Starting failure trace generation...")
    
    # Ensure data files exist
    if not Path(DATA_DIR / "failure_modes.json").exists():
        print("Error: data/failure_modes.json not found!")
        return
    
    if not Path(DATA_DIR / "customer_personas.json").exists():
        print("Error: data/customer_personas.json not found!")
        return
    
    # Initialize generator
    generator = FailureTraceGenerator()
    
    # Generate traces
    traces = generator.generate_all_traces()
    
    if not traces:
        print("No traces generated successfully!")
        return
    
    # Save results
    generator.save_traces(traces, OUTPUT_FILE)
    
    print("Failure trace generation complete!")

if __name__ == "__main__":
    main() 