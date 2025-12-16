import pandas as pd
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from preprocessing import preprocess_dataframe
from data_manager import split_dataset

def main():
    base_dir = r"H:\The Thesis"
    input_file = os.path.join(base_dir, "cyberbullying_tweets.csv")
    output_dir = os.path.join(base_dir, "data", "splits")
    
    print(f"Loading dataset from {input_file}...")
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print("Starting preprocessing...")
    df_clean = preprocess_dataframe(df)
    
    print("Starting data splitting...")
    split_dataset(df_clean, output_dir=output_dir)
    
    print("Preprocessing and splitting complete.")

if __name__ == "__main__":
    main()
