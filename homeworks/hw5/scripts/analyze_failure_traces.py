#!/usr/bin/env python3
"""
Comprehensive Failure Trace Analysis

This script performs complete analysis of synthetic failure traces including:
- Statistical analysis of failure modes and patterns
- Tool usage analysis and failure rates
- Success vs failure pattern comparison
- Detailed insights and recommendations
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
import sys

# Add the hw5 directory to Python path for imports
SCRIPT_DIR = Path(__file__).parent
HW5_ROOT = SCRIPT_DIR.parent
sys.path.append(str(HW5_ROOT))

from analysis.transition_heatmaps import TransitionAnalyzer

class FailureAnalyzer:
    """Comprehensive analyzer for agent failure traces."""
    
    def __init__(self):
        self.traces = []
        self.metadata = {}
        
    def load_traces(self, traces_file: str):
        """Load conversation traces from JSON file."""
        with open(traces_file, 'r') as f:
            data = json.load(f)
            self.traces = data.get("traces", [])
            self.metadata = data.get("metadata", {})
        print(f"Loaded {len(self.traces)} traces for analysis")
    
    def analyze_failure_distribution(self) -> Dict[str, Any]:
        """Analyze distribution of failure modes and categories."""
        distribution = {
            "failure_modes": defaultdict(int),
            "failure_categories": defaultdict(int),
            "personas": defaultdict(int),
            "overall_success_rate": 0,
            "recovery_success_rate": 0
        }
        
        successful_traces = 0
        recovery_attempts = 0
        successful_recoveries = 0
        
        for trace in self.traces:
            # Count failure modes and categories
            distribution["failure_modes"][trace["failure_mode"]] += 1
            distribution["failure_categories"][trace["failure_category"]] += 1
            distribution["personas"][trace["customer_persona"]] += 1
            
            # Track success rates
            if trace["overall_success"]:
                successful_traces += 1
            
            # Track recovery rates
            if trace.get("recovery_success") is not None:
                recovery_attempts += 1
                if trace["recovery_success"]:
                    successful_recoveries += 1
        
        # Calculate rates
        total_traces = len(self.traces)
        distribution["overall_success_rate"] = successful_traces / total_traces if total_traces > 0 else 0
        distribution["recovery_success_rate"] = successful_recoveries / recovery_attempts if recovery_attempts > 0 else 0
        
        # Convert to regular dicts
        distribution["failure_modes"] = dict(distribution["failure_modes"])
        distribution["failure_categories"] = dict(distribution["failure_categories"])
        distribution["personas"] = dict(distribution["personas"])
        
        return distribution
    
    def analyze_tool_usage_patterns(self) -> Dict[str, Any]:
        """Analyze how tools are used and where they fail."""
        tool_analysis = {
            "tool_usage_frequency": defaultdict(int),
            "tool_failure_rates": defaultdict(lambda: {"total": 0, "failures": 0}),
            "tool_by_persona": defaultdict(lambda: defaultdict(int)),
            "tool_chains": defaultdict(int),
            "failure_by_tool": defaultdict(list)
        }
        
        for trace in self.traces:
            persona = trace["customer_persona"]
            
            # Analyze tool usage in this trace
            tool_sequence = []
            for msg in trace["messages"]:
                if msg["role"] == "tool" and msg.get("tool_name"):
                    tool_name = msg["tool_name"]
                    tool_analysis["tool_usage_frequency"][tool_name] += 1
                    tool_analysis["tool_by_persona"][persona][tool_name] += 1
                    tool_sequence.append(tool_name)
                    
                    # Check if this tool call failed
                    tool_analysis["tool_failure_rates"][tool_name]["total"] += 1
                    
                    if msg.get("tool_output", {}).get("error") or msg.get("failure_indicators"):
                        tool_analysis["tool_failure_rates"][tool_name]["failures"] += 1
                        tool_analysis["failure_by_tool"][tool_name].append({
                            "trace_id": trace["trace_id"],
                            "failure_mode": trace["failure_mode"],
                            "error_type": msg.get("tool_output", {}).get("error", "unknown")
                        })
            
            # Record tool chains (sequences of tool usage)
            if len(tool_sequence) > 1:
                chain = "→".join(tool_sequence)
                tool_analysis["tool_chains"][chain] += 1
        
        # Calculate failure rates
        for tool, stats in tool_analysis["tool_failure_rates"].items():
            if stats["total"] > 0:
                stats["failure_rate"] = stats["failures"] / stats["total"]
            else:
                stats["failure_rate"] = 0
        
        # Convert to regular dicts
        tool_analysis["tool_usage_frequency"] = dict(tool_analysis["tool_usage_frequency"])
        tool_analysis["tool_failure_rates"] = dict(tool_analysis["tool_failure_rates"])
        tool_analysis["tool_by_persona"] = {k: dict(v) for k, v in tool_analysis["tool_by_persona"].items()}
        tool_analysis["tool_chains"] = dict(tool_analysis["tool_chains"])
        tool_analysis["failure_by_tool"] = dict(tool_analysis["failure_by_tool"])
        
        return tool_analysis
    
    def analyze_conversation_characteristics(self) -> Dict[str, Any]:
        """Analyze conversation characteristics and patterns."""
        characteristics = {
            "length_distribution": {
                "by_success": {"success": [], "failure": []},
                "by_category": defaultdict(list),
                "by_persona": defaultdict(list)
            },
            "message_timing": {
                "average_response_time": 0,
                "timing_by_role": defaultdict(list)
            },
            "conversation_complexity": {
                "tool_calls_per_conversation": [],
                "unique_tools_per_conversation": [],
                "role_switches": []
            }
        }
        
        total_response_times = []
        
        for trace in self.traces:
            conversation_length = len(trace["messages"])
            
            # Length analysis
            success_category = "success" if trace["overall_success"] else "failure"
            characteristics["length_distribution"]["by_success"][success_category].append(conversation_length)
            characteristics["length_distribution"]["by_category"][trace["failure_category"]].append(conversation_length)
            characteristics["length_distribution"]["by_persona"][trace["customer_persona"]].append(conversation_length)
            
            # Complexity analysis
            tool_calls = sum(1 for msg in trace["messages"] if msg["role"] == "tool")
            unique_tools = len(set(msg.get("tool_name") for msg in trace["messages"] if msg.get("tool_name")))
            
            characteristics["conversation_complexity"]["tool_calls_per_conversation"].append(tool_calls)
            characteristics["conversation_complexity"]["unique_tools_per_conversation"].append(unique_tools)
            
            # Role switching analysis
            roles = [msg["role"] for msg in trace["messages"]]
            role_switches = sum(1 for i in range(1, len(roles)) if roles[i] != roles[i-1])
            characteristics["conversation_complexity"]["role_switches"].append(role_switches)
            
            # Timing analysis (simplified - using message order as proxy)
            for i in range(1, len(trace["messages"])):
                # In real implementation, would parse timestamps
                total_response_times.append(5.0)  # Placeholder
        
        # Calculate averages
        if total_response_times:
            characteristics["message_timing"]["average_response_time"] = np.mean(total_response_times)
        
        # Convert defaultdicts to regular dicts
        characteristics["length_distribution"]["by_category"] = dict(characteristics["length_distribution"]["by_category"])
        characteristics["length_distribution"]["by_persona"] = dict(characteristics["length_distribution"]["by_persona"])
        
        return characteristics
    
    def identify_failure_patterns(self) -> Dict[str, Any]:
        """Identify common patterns that lead to failures."""
        patterns = {
            "common_failure_sequences": defaultdict(int),
            "failure_triggers": defaultdict(int),
            "persona_specific_failures": defaultdict(lambda: defaultdict(int)),
            "tool_failure_cascades": [],
            "early_warning_indicators": defaultdict(int)
        }
        
        for trace in self.traces:
            persona = trace["customer_persona"]
            failure_mode = trace["failure_mode"]
            
            # Count persona-specific failures
            patterns["persona_specific_failures"][persona][failure_mode] += 1
            
            # Analyze message sequences for failure patterns
            roles = [msg["role"] for msg in trace["messages"]]
            failure_messages = [i for i, msg in enumerate(trace["messages"]) if msg.get("failure_indicators")]
            
            if failure_messages:
                # Look at sequences leading to failure
                for failure_idx in failure_messages:
                    if failure_idx >= 2:  # Need at least 3 messages for pattern
                        pattern = "→".join(roles[failure_idx-2:failure_idx+1])
                        patterns["common_failure_sequences"][pattern] += 1
                    
                    # Record failure triggers
                    failure_msg = trace["messages"][failure_idx]
                    if failure_msg.get("failure_indicators"):
                        for indicator in failure_msg["failure_indicators"]:
                            patterns["failure_triggers"][indicator] += 1
            
            # Look for tool failure cascades
            tool_failures = []
            for i, msg in enumerate(trace["messages"]):
                if msg["role"] == "tool" and (msg.get("tool_output", {}).get("error") or msg.get("failure_indicators")):
                    tool_failures.append((i, msg.get("tool_name")))
            
            if len(tool_failures) > 1:
                patterns["tool_failure_cascades"].append({
                    "trace_id": trace["trace_id"],
                    "failures": tool_failures,
                    "cascade_length": len(tool_failures)
                })
            
            # Early warning indicators (patterns in first 3 messages that predict failure)
            if len(trace["messages"]) >= 3 and not trace["overall_success"]:
                early_pattern = "→".join(roles[:3])
                patterns["early_warning_indicators"][early_pattern] += 1
        
        # Convert to regular dicts
        patterns["common_failure_sequences"] = dict(patterns["common_failure_sequences"])
        patterns["failure_triggers"] = dict(patterns["failure_triggers"])
        patterns["persona_specific_failures"] = {k: dict(v) for k, v in patterns["persona_specific_failures"].items()}
        patterns["early_warning_indicators"] = dict(patterns["early_warning_indicators"])
        
        return patterns
    
    def generate_insights_and_recommendations(self, 
                                            distribution: Dict[str, Any],
                                            tool_analysis: Dict[str, Any], 
                                            patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Generate actionable insights and recommendations."""
        insights = {
            "key_findings": [],
            "critical_issues": [],
            "recommendations": [],
            "success_factors": [],
            "monitoring_metrics": []
        }
        
        # Key findings
        if distribution["failure_modes"]:
            most_common_failure = max(distribution["failure_modes"].items(), key=lambda x: x[1])
            insights["key_findings"].append(f"Most common failure mode: {most_common_failure[0]} ({most_common_failure[1]} occurrences)")
        
        if distribution["overall_success_rate"] < 0.8:
            insights["key_findings"].append(f"Low overall success rate: {distribution['overall_success_rate']:.1%}")
        
        # Tool analysis insights
        if tool_analysis["tool_failure_rates"]:
            worst_tool = max(tool_analysis["tool_failure_rates"].items(), 
                           key=lambda x: x[1]["failure_rate"])
            if worst_tool[1]["failure_rate"] > 0.3:
                insights["critical_issues"].append(f"High failure rate for {worst_tool[0]}: {worst_tool[1]['failure_rate']:.1%}")
        
        # Pattern insights
        if patterns["common_failure_sequences"]:
            most_common_pattern = max(patterns["common_failure_sequences"].items(), key=lambda x: x[1])
            insights["key_findings"].append(f"Most common failure sequence: {most_common_pattern[0]}")
        
        # Recommendations
        insights["recommendations"] = [
            "Implement better error handling for tool failures",
            "Add validation for tool input parameters",
            "Improve agent reasoning when tools return empty results",
            "Add recovery strategies for common failure patterns",
            "Implement user-friendly error explanations"
        ]
        
        # Success factors
        if distribution["recovery_success_rate"] > 0.6:
            insights["success_factors"].append("Good recovery success rate indicates resilient design")
        
        # Monitoring metrics
        insights["monitoring_metrics"] = [
            "Tool failure rates by tool type",
            "Conversation success rate by persona",
            "Average recovery time after failures",
            "Frequency of specific failure modes"
        ]
        
        return insights
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate complete failure analysis report."""
        print("Generating comprehensive failure analysis...")
        
        # Run all analyses
        distribution = self.analyze_failure_distribution()
        tool_analysis = self.analyze_tool_usage_patterns()
        characteristics = self.analyze_conversation_characteristics()
        patterns = self.identify_failure_patterns()
        insights = self.generate_insights_and_recommendations(distribution, tool_analysis, patterns)
        
        # Generate transition analysis
        transition_analyzer = TransitionAnalyzer()
        transition_analyzer.traces = self.traces
        transition_results = transition_analyzer.analyze_failure_traces()
        
        # Compile complete report
        report = {
            "analysis_metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_traces_analyzed": len(self.traces),
                "analysis_version": "1.0"
            },
            "failure_distribution": distribution,
            "tool_analysis": tool_analysis,
            "conversation_characteristics": characteristics,
            "failure_patterns": patterns,
            "transition_analysis": transition_results,
            "insights_and_recommendations": insights
        }
        
        return report
    
    def save_report(self, report: Dict[str, Any], output_path: str):
        """Save analysis report to JSON file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Saved comprehensive analysis report to {output_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("FAILURE ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"Total traces analyzed: {report['analysis_metadata']['total_traces_analyzed']}")
        print(f"Overall success rate: {report['failure_distribution']['overall_success_rate']:.1%}")
        print(f"Recovery success rate: {report['failure_distribution']['recovery_success_rate']:.1%}")
        
        if report['failure_distribution']['failure_categories']:
            print(f"\nMost common failure category: {max(report['failure_distribution']['failure_categories'].items(), key=lambda x: x[1])}")
        
        if report['tool_analysis']['tool_failure_rates']:
            print(f"Most problematic tool: {max(report['tool_analysis']['tool_failure_rates'].items(), key=lambda x: x[1]['failure_rate'])}")
        
        print(f"\nKey insights:")
        for insight in report['insights_and_recommendations']['key_findings'][:3]:
            print(f"  • {insight}")
        
        print(f"\nCritical issues:")
        for issue in report['insights_and_recommendations']['critical_issues'][:3]:
            print(f"  • {issue}")

def main():
    """Main execution function."""
    # Path resolution - get hw5 root directory from script location
    SCRIPT_DIR = Path(__file__).parent
    HW5_ROOT = SCRIPT_DIR.parent
    DATA_DIR = HW5_ROOT / "data"
    RESULTS_DIR = HW5_ROOT / "results"
    
    # Check if traces file exists
    traces_file = DATA_DIR / "synthetic_traces.json"
    if not traces_file.exists():
        print(f"Error: {traces_file} not found!")
        print("Please run generate_failure_traces.py first")
        return
    
    # Initialize analyzer
    analyzer = FailureAnalyzer()
    analyzer.load_traces(str(traces_file))
    
    # Generate comprehensive report
    report = analyzer.generate_comprehensive_report()
    
    # Save report
    analyzer.save_report(report, str(RESULTS_DIR / "failure_analysis.json"))
    
    print(f"\nAnalysis complete! Check {RESULTS_DIR / 'failure_analysis.json'} for detailed results.")
    print(f"Visualizations saved in {RESULTS_DIR / 'visualizations'}/")

if __name__ == "__main__":
    main() 