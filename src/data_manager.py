import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
import os

def split_dataset(df, target_col='cyberbullying_type', output_dir='data/splits', seed=42):
    """
    Splits the dataset into:
    1. Train/Test (75/25)
    2. Low-Data Train (10% of Train)
    3. 5-Fold CV on Train
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Splitting dataset with shape: {df.shape}")
    
    # 75/25 Stratified Split
    train, test = train_test_split(
        df, 
        test_size=0.25, 
        stratify=df[target_col], 
        random_state=seed
    )
    
    # Save full splits
    train.to_csv(os.path.join(output_dir, 'train_full.csv'), index=False)
    test.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    print(f"Saved train_full.csv ({len(train)}) and test.csv ({len(test)})")
    
    # 10% Low-Data Regime (stratified subset of train)
    train_low, _ = train_test_split(
        train,
        train_size=0.10,
        stratify=train[target_col],
        random_state=seed
    )
    train_low.to_csv(os.path.join(output_dir, 'train_low_10pct.csv'), index=False)
    print(f"Saved train_low_10pct.csv ({len(train_low)})")
    
    # 5-Fold CV on Train Full
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (train_idx, val_idx) in enumerate(skf.split(train, train[target_col])):
        fold_train = train.iloc[train_idx]
        fold_val = train.iloc[val_idx]
        
        fold_dir = os.path.join(output_dir, f'fold_{fold}')
        os.makedirs(fold_dir, exist_ok=True)
        fold_train.to_csv(os.path.join(fold_dir, 'train.csv'), index=False)
        fold_val.to_csv(os.path.join(fold_dir, 'val.csv'), index=False)
        print(f"  Saved Fold {fold}: Train ({len(fold_train)}), Val ({len(fold_val)})")

    print(f"All splits saved to {output_dir}")
