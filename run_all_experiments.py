import subprocess
import os
import sys

def main():
    python_exe = sys.executable
    script_path = os.path.join("H:\\The Thesis", "run_llm_experiments.py")
    
    # Define the experimental matrix
    models = [
        "Qwen/Qwen2.5-7B-Instruct",
        "deepseek-ai/deepseek-llm-7b-chat",
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3"
    ]
    
    strategies = ["neutral", "aggressive"]
    shots_list = [0, 4, 8]
    
    print("--- Starting Full Experimental Suite (PRODUCTION MODE) ---")
    print("WARNING: This will process the entire test set for all configurations.")
    print("Ensure you have sufficient VRAM and time.")
    
    for model in models:
        for strategy in strategies:
            for shots in shots_list:
                print(f"\nRunning: Model={model}, Strategy={strategy}, Shots={shots}")
                
                cmd = [
                    python_exe, 
                    script_path,
                    "--model", model,
                    "--strategy", strategy,
                    "--shots", str(shots),
                    "--limit", "0"  # 0 means NO LIMIT (Full Dataset)
                ]
                
                try:
                    subprocess.check_call(cmd)
                except subprocess.CalledProcessError as e:
                    print(f"Failed: {model} | {strategy} | {shots} - Error: {e}")
                    continue

    print("\n--- All Experiments Completed ---")

if __name__ == "__main__":
    main()
