import random
import litellm
from litellm import completion, model_cost, Cache
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from typing import Tuple
from dotenv import load_dotenv

# Set up caching and environment
litellm.cache = Cache(type="disk")
load_dotenv()
random.seed(42)

TARGET_ACCURACY = 0.99

def cost_given_token_breakdown(model: str, input_tokens_not_cached: int, input_tokens_cached: int, output_tokens: int) -> float:
    input_cost_per_token = model_cost[model]["input_cost_per_token"]
    output_cost_per_token = model_cost[model]["output_cost_per_token"]
    input_cost_per_cached_token = model_cost[model]["cache_read_input_token_cost"]
    
    return input_cost_per_token * input_tokens_not_cached + input_cost_per_cached_token * input_tokens_cached + output_cost_per_token * output_tokens
  
def cost_of_completion(response) -> float:
    model = response.model
    return cost_given_token_breakdown(model, response.usage["prompt_tokens"], 0, response.usage["completion_tokens"])

def get_answer_prob_binary(logprobs_dict, answer):
    # Convert logprobs to probabilities
    probs = {token: np.exp(logprob) for token, logprob in logprobs_dict.items()}
    
    # Check if both True and False are in the tokens
    if 'True' in probs and 'False' in probs:
        true_prob = probs['True']
        false_prob = probs['False']
        # Normalize
        answer_prob = true_prob if answer == 1 else false_prob
        
        return answer_prob / (true_prob + false_prob)
    
    # Return the max probability
    return max(probs.values())

def process_doc(model: str, text: str) -> Tuple[int, float, float]:
    prompt = f"""I will give you an SMS text message. Here is the message: {text}

Your task is to determine if this message is either legitimate or harmless spam.

- True if the message is EITHER:
  - A legitimate message from a real person or business
  - Harmless spam without financial risk
- False if the message contains anything financially risky

You must respond with ONLY True or False:"""

    try:
        res = completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            logprobs=True,
            top_logprobs=10,
            max_tokens=1,
            num_retries=5,
            caching=True,
            temperature=0.0,
            timeout=10,
        )
        
        response = res.choices[0].message.content
        response_converted = 1 if response.lower() == "true" else 0
        
        # Get confidence only for proxy model
        if model == "gpt-4o-mini":
            first_logprob = res.choices[0].logprobs['content'][0]
            confidence = get_answer_prob_binary({item.token: item.logprob for item in first_logprob.top_logprobs}, response_converted)
        else:
            confidence = 0.0  # No confidence for oracle
            
        cost = cost_of_completion(res)
        
        return response_converted, confidence, cost
    except Exception as e:
        print(f"Error processing message with {model}, {e}")
        import traceback
        traceback.print_exc()
        return 0, 0.0, 0.0

def load_data(file_path: str, limit: int = 500) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load data and return train and test DataFrames"""
    df = pd.read_csv(file_path)
    data = df["text"].tolist()
    
    # Shuffle the data
    random.shuffle(data)
    data = data[:limit]
    
    # Split into training and test sets
    split_idx = int(len(data) * 0.2)
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    
    # Create DataFrames
    train_df = pd.DataFrame({"text": train_data})
    test_df = pd.DataFrame({"text": test_data})
    
    return train_df, test_df

def find_thresholds(train_df: pd.DataFrame) -> dict:
    # Will return a dictionary mapping class to the threshold for that class
    
    distinct_classes = train_df["proxy_prediction"].unique()
    thresholds = {}
    
    for c in distinct_classes:
        # Find the threshold for this class that maximizes queries that terminate at the proxy
        # Still needs to meet the target accuracy
        class_predictions = train_df[train_df["proxy_prediction"] == c]
        possible_thresholds = class_predictions["proxy_confidence"].unique().tolist()
        
        # Sort by confidence, ascending
        possible_thresholds.sort()
        
        found_threshold = False
        
        # Find the threshold that maximizes queries that terminate at the proxy
        for t in possible_thresholds:
            # Find the accuracy of the proxy when the threshold is t
            class_predictions_above_threshold = class_predictions[class_predictions["proxy_confidence"] >= t]
            accuracy_at_threshold = (class_predictions_above_threshold["proxy_prediction"] == class_predictions_above_threshold["oracle_prediction"]).mean()
            if accuracy_at_threshold >= TARGET_ACCURACY:
                thresholds[c] = t
                found_threshold = True
                break
        
        if not found_threshold:
            print(f"No threshold found for class {c}")
            thresholds[c] = float("inf")
    
    return thresholds

def simulate_cascade(test_df: pd.DataFrame, thresholds: dict) -> None:
    """
    Simulate the cascade model on test data and calculate cost savings.
    
    Args:
        test_df: DataFrame with proxy and oracle predictions and costs
        thresholds: Dictionary mapping class predictions to confidence thresholds
    """
    cascade_predictions = []
    cascade_costs = []
    uses_oracle = []
    
    for _, row in test_df.iterrows():
        proxy_pred = row['proxy_prediction']
        proxy_conf = row['proxy_confidence']
        proxy_cost = row['proxy_cost']
        oracle_pred = row['oracle_prediction']
        oracle_cost = row['oracle_cost']
        
        # Get threshold for this prediction class
        threshold = thresholds.get(proxy_pred, float('inf'))
        
        # If proxy confidence is above threshold, use proxy
        if proxy_conf >= threshold:
            cascade_predictions.append(proxy_pred)
            cascade_costs.append(proxy_cost)  # Only pay proxy cost
            uses_oracle.append(False)
        else:
            # Use oracle (but still pay for proxy since we had to run it first)
            cascade_predictions.append(oracle_pred)
            cascade_costs.append(proxy_cost + oracle_cost)  # Pay both costs
            uses_oracle.append(True)
    
    # Calculate metrics
    total_cascade_cost = sum(cascade_costs)
    total_proxy_only_cost = test_df['proxy_cost'].sum()
    total_oracle_only_cost = test_df['oracle_cost'].sum()
    total_both_models_cost = total_proxy_only_cost + total_oracle_only_cost
    
    oracle_usage_rate = sum(uses_oracle) / len(uses_oracle)
    proxy_termination_rate = 1 - oracle_usage_rate
    
    # Calculate accuracy (assuming oracle predictions are ground truth)
    cascade_accuracy = sum(cp == op for cp, op in zip(cascade_predictions, test_df['oracle_prediction'])) / len(cascade_predictions)
    
    # Print results
    print(f"\n=== CASCADE SIMULATION RESULTS ===")
    print(f"Total samples: {len(test_df)}")
    print(f"Proxy termination rate: {proxy_termination_rate:.2%}")
    print(f"Oracle usage rate: {oracle_usage_rate:.2%}")
    print(f"Cascade accuracy: {cascade_accuracy:.4f}")
    print(f"Target accuracy: {TARGET_ACCURACY}")
    print(f"")
    print(f"=== COST ANALYSIS ===")
    print(f"Cascade total cost: ${total_cascade_cost:.4f}")
    print(f"Proxy-only cost: ${total_proxy_only_cost:.4f}")
    print(f"Oracle-only cost: ${total_oracle_only_cost:.4f}")
    
    return {
        'cascade_predictions': cascade_predictions,
        'cascade_costs': cascade_costs,
        'uses_oracle': uses_oracle,
        'total_cost': total_cascade_cost,
        'oracle_usage_rate': oracle_usage_rate,
        'accuracy': cascade_accuracy
    }
    
    


def main():
    # Load data
    train_df, test_df = load_data("lesson-8/sms_spam.csv")
    
    print(f"Loaded {len(train_df)} training samples and {len(test_df)} test samples")
    
    # Run predictions with both models
    train_proxy_results = []
    train_oracle_results = []
    test_proxy_results = []
    test_oracle_results = []
    
    with ThreadPoolExecutor(max_workers=32) as executor:
        # Run train proxy predictions
        print("Running train proxy (gpt-4o-mini) predictions...")
        train_proxy_futures = [
            executor.submit(process_doc, "gpt-4o-mini", text) 
            for text in train_df["text"].tolist()
        ]
        
        # Run train oracle predictions
        print("Running train oracle (gpt-4o) predictions...")
        train_oracle_futures = [
            executor.submit(process_doc, "gpt-4o", text) 
            for text in train_df["text"].tolist()
        ]
        
        # Run test proxy predictions
        print("Running test proxy (gpt-4o-mini) predictions...")
        test_proxy_futures = [
            executor.submit(process_doc, "gpt-4o-mini", text) 
            for text in test_df["text"].tolist()
        ]
        
        # Run test oracle predictions
        print("Running test oracle (gpt-4o) predictions...")
        test_oracle_futures = [
            executor.submit(process_doc, "gpt-4o", text) 
            for text in test_df["text"].tolist()
        ]
        
        # Collect train proxy results
        for future in tqdm(train_proxy_futures, desc="Collecting train proxy results"):
            train_proxy_results.append(future.result())
            
        # Collect train oracle results
        for future in tqdm(train_oracle_futures, desc="Collecting train oracle results"):
            train_oracle_results.append(future.result())
            
        # Collect test proxy results
        for future in tqdm(test_proxy_futures, desc="Collecting test proxy results"):
            test_proxy_results.append(future.result())
            
        # Collect test oracle results
        for future in tqdm(test_oracle_futures, desc="Collecting test oracle results"):
            test_oracle_results.append(future.result())
    
    # Create final DataFrame with requested columns
    train_df = pd.DataFrame({
        'text': train_df['text'],
        'proxy_prediction': [result[0] for result in train_proxy_results],
        'proxy_confidence': [result[1] for result in train_proxy_results],
        'proxy_cost': [result[2] for result in train_proxy_results],
        'oracle_prediction': [result[0] for result in train_oracle_results],
        'oracle_cost': [result[2] for result in train_oracle_results]
    })
    test_df = pd.DataFrame({
        'text': test_df['text'],
        'proxy_prediction': [result[0] for result in test_proxy_results],
        'proxy_confidence': [result[1] for result in test_proxy_results],
        'proxy_cost': [result[2] for result in test_proxy_results],
        'oracle_prediction': [result[0] for result in test_oracle_results],
        'oracle_cost': [result[2] for result in test_oracle_results]
    })
    
    # Print accuracy
    print(f"Proxy accuracy: {(train_df['proxy_prediction'] == train_df['oracle_prediction']).mean()}")
    
    # Find thresholds from train data
    train_thresholds = find_thresholds(train_df)
    print(f"Thresholds found: {train_thresholds}")
    
    # Simulate cascade on test data
    cascade_results = simulate_cascade(test_df, train_thresholds)
    
    # Save results
    train_df.to_csv("lesson-8/sms_spam_predictions_train.csv", index=False)
    test_df.to_csv("lesson-8/sms_spam_predictions_test.csv", index=False)
    
    print(f"\nResults saved to sms_spam_predictions_train.csv and sms_spam_predictions_test.csv")
    print(f"Total train proxy cost: ${train_df['proxy_cost'].sum():.4f}")
    print(f"Total train oracle cost: ${train_df['oracle_cost'].sum():.4f}")
    print(f"Total test proxy cost: ${test_df['proxy_cost'].sum():.4f}")
    print(f"Total test oracle cost: ${test_df['oracle_cost'].sum():.4f}")
    print("\nFirst few train predictions:")
    print(train_df[['text', 'proxy_prediction', 'proxy_confidence', 'oracle_prediction']].head())

if __name__ == "__main__":
    main()