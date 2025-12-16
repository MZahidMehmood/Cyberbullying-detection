import sys
import os
import argparse

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from llm_core import LLMPipeline, unload_model
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Gemma Replication (RANLP-2025)")
    parser.add_argument("--model", type=str, default="google/gemma-7b-it", help="Gemma model path")
    parser.add_argument("--limit", type=int, default=0, help="Limit samples")
    args = parser.parse_args()

    print(f"Starting Gemma Replication with model: {args.model}")
    
    # Load Data
    data_dir = os.path.join("data", "splits")
    try:
        test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
        train_df = pd.read_csv(os.path.join(data_dir, 'train_full.csv'))
    except FileNotFoundError:
        print("Data splits not found.")
        return

    if args.limit > 0:
        test_df = test_df.head(args.limit)

    # Initialize Pipeline
    pipeline = LLMPipeline(args.model)

    # Run with Neutral Prompt (Standard for replication unless specified otherwise)
    results = pipeline.run_batch(
        texts=test_df['cleaned_text'].tolist(),
        strategy="neutral",
        train_df=train_df,
        n_shots=0 # Zero-shot as per typical baseline, or adjust as needed
    )
    
    # Save Results
    output_dir = os.path.join("results", "gemma_replication")
    os.makedirs(output_dir, exist_ok=True)
    
    output_df = test_df.copy()
    output_df['pred_label'] = [r.get('label', 'error') for r in results]
    output_df['confidence'] = [r.get('confidence', 0.0) for r in results]
    output_df.to_csv(os.path.join(output_dir, "gemma_results.csv"), index=False)
    print("Gemma replication complete.")
    
    unload_model(pipeline)

if __name__ == "__main__":
    main()
