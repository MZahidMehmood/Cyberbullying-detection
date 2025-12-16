import os
import subprocess
import sys

def run_command(command, description):
    print(f"\n--- {description} ---")
    print(f"Running: {command}")
    try:
        subprocess.check_call(command, shell=True)
        print("Success.")
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")

def main():
    python_exe = sys.executable
    base_dir = r"H:\The Thesis"
    
    print("Cyberbullying Detection Project - Master Runner")
    print("1. Preprocessing (Data Cleaning & Splitting)")
    print("2. Classical Baselines (Train & Eval)")
    print("3. LLM Dry Run (Test with small model & limit)")
    print("4. FULL LLM SUITE (Production - All Models/Shots, No Limit)")
    print("5. Evaluation & Reporting (Generate Plots/Tables)")
    
    choice = input("Enter choice (1-5): ")
    
    if choice == '1':
        run_command(f'{python_exe} "{os.path.join(base_dir, "run_preprocessing.py")}"', "Preprocessing")
        
    elif choice == '2':
        run_command(f'{python_exe} "{os.path.join(base_dir, "run_baselines.py")}"', "Classical Baselines")
        
    elif choice == '3':
        run_command(f'{python_exe} "{os.path.join(base_dir, "run_llm_experiments.py")}" --limit 5 --model Qwen/Qwen2.5-0.5B-Instruct', "LLM Dry Run")
        
    elif choice == '4':
        print("Launching Full Experimental Suite...")
        run_command(f'{python_exe} "{os.path.join(base_dir, "run_all_experiments.py")}"', "Full LLM Production Run")
        
    elif choice == '5':
        print("Generating Artifacts from Results...")
        run_command(f'{python_exe} "{os.path.join(base_dir, "src", "reporting.py")}"', "Reporting")

if __name__ == "__main__":
    main()
