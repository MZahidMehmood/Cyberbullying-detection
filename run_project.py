import subprocess
import os
import sys

def run_script(script_name):
    print(f"\n{'='*60}")
    print(f"RUNNING: {script_name}")
    print(f"{'='*60}")
    try:
        subprocess.check_call([sys.executable, script_name])
        print(f"SUCCESS: {script_name}")
    except subprocess.CalledProcessError as e:
        print(f"FAILURE: {script_name} failed with error {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("Starting Cyberbullying Detection Project - Full Pipeline")
    
    # 1. Preprocessing (Raw -> Clean)
    run_script(os.path.join("src", "preprocessing.py"))
    
    # 2. Splits (Clean -> Train/Test/Folds)
    run_script(os.path.join("src", "create_splits.py"))
    
    # 3. Baselines (TF-IDF + ML Models)
    run_script(os.path.join("src", "baselines.py"))
    
    # 4. LLM Experiments (Full Suite: Models x Strategies x Shots x Regimes)
    run_script("run_all_experiments.py")
    
    # 5. Reporting & Artifacts (Pareto, Confusion Matrix, etc.)
    run_script(os.path.join("src", "reporting.py"))
    
    # 6. Appendix: Gemma Replication
    run_script(os.path.join("src", "experiments", "gemma_replication.py"))
    
    # 6. Appendix: LoRA Ablation
    run_script(os.path.join("src", "experiments", "lora_ablation.py"))
    
    print("\n" + "="*60)
    print("PROJECT PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60)
