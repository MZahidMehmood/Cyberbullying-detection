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
    output_df['cost_est'] = 0.0 # Placeholder for local inference cost (electricity/hardware amortized)
    
    output_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    # 1. Save Prompt Artifact
    prompt_file = output_file.replace('.csv', '_prompt.txt')
    with open(prompt_file, 'w', encoding='utf-8') as f:
        # Reconstruct a sample prompt to save as artifact
        sample_prompt = pipeline.construct_prompt("SAMPLE TWEET TEXT", strategy, [], cues)
        f.write(sample_prompt)
        
    # 2. Generate Checksum
    import hashlib
    with open(output_file, 'rb') as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    
    checksum_file = output_file.replace('.csv', '_checksum.sha256')
    with open(checksum_file, 'w') as f:
        f.write(file_hash)
        
    # 3. Generate Model Card (Experiment Report)
    card_file = output_file.replace('.csv', '_model_card.md')
    with open(card_file, 'w') as f:
        f.write(f"# Model Card: {model_name}\n\n")
        f.write(f"## Experiment Details\n")
        f.write(f"- **Strategy**: {strategy}\n")
        f.write(f"- **Shots**: {n_shots}\n")
        f.write(f"- **Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **Device**: {pipeline.device}\n\n")
        f.write(f"## Performance\n")
        f.write(f"- **Samples Processed**: {len(output_df)}\n")
        f.write(f"- **Avg Latency**: {output_df['latency'].mean():.4f}s\n")
        f.write(f"- **Avg Input Tokens**: {output_df['input_tokens'].mean():.1f}\n")
        f.write(f"- **Avg Output Tokens**: {output_df['output_tokens'].mean():.1f}\n")
        
    print(f"Artifacts generated: Prompt ({prompt_file}), Checksum ({checksum_file}), Model Card ({card_file})")
    
    # Cleanup
    unload_model(pipeline)

if __name__ == "__main__":
    main()
