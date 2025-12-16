import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold, train_test_split

def create_splits(input_file, output_dir, seed=42):
    """
    Generates:
    1. 75/25 Stratified Train/Test Split
    2. 5-Fold Stratified CV on the Training Set
    """
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Train/Test Split (75/25)
    print("Generating 75/25 Stratified Train/Test Split...")
    train_df, test_df = train_test_split(
        df, 
        test_size=0.25, 
        stratify=df['cyberbullying_type'], 
        random_state=seed
    )
    
    train_df.to_csv(os.path.join(output_dir, 'train_full.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    print(f"Saved train_full.csv ({len(train_df)}) and test.csv ({len(test_df)})")
    
    # 2. 5-Fold CV on Train
    print("Generating 5-Fold CV Splits for Training Set...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['cyberbullying_type'])):
        fold_train = train_df.iloc[train_idx]
        fold_val = train_df.iloc[val_idx]
        
        fold_dir = os.path.join(output_dir, f'fold_{fold}')
        os.makedirs(fold_dir, exist_ok=True)
        
        fold_train.to_csv(os.path.join(fold_dir, 'train.csv'), index=False)
        fold_val.to_csv(os.path.join(fold_dir, 'val.csv'), index=False)
        print(f"  Saved Fold {fold}: Train ({len(fold_train)}), Val ({len(fold_val)})")

if __name__ == "__main__":
    # Assuming preprocessing has run and saved a clean file
    # If not, we might need to integrate this into preprocessing.py or run after.
    # For now, we assume 'data/processed/clean_data.csv' exists or we use the raw file if clean.
    # Let's assume we run this on the raw dataset after cleaning.
    
    # Path to the CLEANED dataset (output of preprocessing)
    # Since preprocessing.py returns a df but doesn't save it in the snippet I saw,
    # I should probably update preprocessing.py to save 'clean_data.csv' first.
    # Or I can make this script load raw, clean, then split.
    # Let's make this script standalone but dependent on a cleaned file.
    
    input_path = os.path.join("data", "processed", "clean_data.csv")
    output_path = os.path.join("data", "splits")
    
    if os.path.exists(input_path):
        create_splits(input_path, output_path)
    else:
        print(f"Input file {input_path} not found. Please run preprocessing first.")
