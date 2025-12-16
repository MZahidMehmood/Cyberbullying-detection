import sys
import os
import pandas as pd
import argparse

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from llm_core import LLMPipeline, unload_model

def main():
    parser = argparse.ArgumentParser(description="Run LLM Experiments")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="HuggingFace model path")
    parser.add_argument("--strategy", type=str, choices=["neutral", "aggressive"], default="neutral")
    parser.add_argument("--shots", type=int, default=0, help="Number of few-shot examples")
    parser.add_argument("--limit", type=int, default=10, help="Limit number of test samples for dry run")
    args = parser.parse_args()

    print(f"Starting Experiment: Model={args.model}, Strategy={args.strategy}, Shots={args.shots}")
    
    # Load Data
    data_dir = r"H:\The Thesis\data\splits"
    try:
        test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
        train_df = pd.read_csv(os.path.join(data_dir, 'train_full.csv'))
    except FileNotFoundError:
        print("Data splits not found. Run preprocessing first.")
        return

    # Limit for testing
    if args.limit > 0:
        test_df = test_df.head(args.limit)
        print(f"Limiting test set to first {args.limit} samples.")

    # Initialize Pipeline
    try:
        pipeline = LLMPipeline(args.model)
    except Exception as e:
        print(f"Failed to initialize pipeline: {e}")
        return

    # Run Inference
    results = pipeline.run_batch(
        texts=test_df['cleaned_text'].tolist(),
        strategy=args.strategy,
        train_df=train_df,
        n_shots=args.shots
    )
    
    # Save Results
    output_dir = r"H:\The Thesis\results\llm_experiments"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f"results_{args.strategy}_{args.shots}shot.csv")
    
    # Combine inputs with results
    output_df = test_df.copy()
    output_df['pred_label'] = [r.get('label', 'error') for r in results]
    output_df['confidence'] = [r.get('confidence', 0.0) for r in results]
    output_df['rationale'] = [r.get('rationale', '') for r in results]
    output_df['latency'] = [r.get('latency', 0.0) for r in results]
    output_df['input_tokens'] = [r.get('input_tokens', 0) for r in results]
    output_df['output_tokens'] = [r.get('output_tokens', 0) for r in results]
    output_df['vram_mb'] = [r.get('vram_mb', 0.0) for r in results]
    
    output_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    # Cleanup
    unload_model(pipeline)

if __name__ == "__main__":
    main()
