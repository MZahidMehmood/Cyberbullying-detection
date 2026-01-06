import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import resample

class DQEAugmenter:
    """
    Diversity Query Expansion (DQE) Augmenter.
    
    Since we cannot query the web for new data, this class implements 'Semantic Augmentation':
    1. Identifies minority classes.
    2. Uses embeddings (or TF-IDF) to find nearest neighbors within the class.
    3. Generates synthetic samples or performs smart oversampling to balance the class distribution.
    
    For this version, we use a simplified approach:
    - Target: Expand minority classes to match the median or majority class size.
    - Method: Random Oversampling (effective and robust for text baselines).
      Future upgrade: Use SBERT to find semantic neighbors and interpolate (SMOTE-like for text).
    """
    
    def __init__(self, target_count='auto', random_state=42):
        self.target_count = target_count
        self.random_state = random_state
        
    def fit_resample(self, df, text_col, label_col):
        """
        Resamples the dataframe to balance classes.
        """
        print(f"DQE: Original distribution:\n{df[label_col].value_counts()}")
        
        # Get class counts
        counts = df[label_col].value_counts()
        majority_count = counts.max()
        
        if self.target_count == 'auto':
            target_n = majority_count
        else:
            target_n = self.target_count
            
        dfs = []
        for label, count in counts.items():
            class_df = df[df[label_col] == label]
            
            if count < target_n:
                # Oversample
                print(f"DQE: Augmenting class '{label}' from {count} to {target_n}...")
                resampled_df = resample(
                    class_df,
                    replace=True,
                    n_samples=target_n,
                    random_state=self.random_state
                )
                dfs.append(resampled_df)
            else:
                # Keep as is (or downsample if strictly balancing, but standard DQE usually expands)
                dfs.append(class_df)
                
        augmented_df = pd.concat(dfs).sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        
        print(f"DQE: Augmented distribution:\n{augmented_df[label_col].value_counts()}")
        return augmented_df

if __name__ == "__main__":
    # Test stub
    data = {
        'text': ['a', 'b', 'c', 'd', 'e'],
        'label': ['min', 'min', 'maj', 'maj', 'maj']
    }
    df = pd.DataFrame(data)
    augmenter = DQEAugmenter()
    aug_df = augmenter.fit_resample(df, 'text', 'label')
