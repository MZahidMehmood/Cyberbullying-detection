import pandas as pd
import numpy as np
import os
import json
import torch
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, f1_score, matthews_corrcoef, average_precision_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sentence_transformers import SentenceTransformer

# Import DQE
try:
    from src.dqe import DQEAugmenter
except ImportError:
    # Handle running as script from root
    import sys
    sys.path.append(os.getcwd())
    from src.dqe import DQEAugmenter

def get_embeddings(texts, model_name='all-MiniLM-L6-v2'):
    """
    Generates SBERT embeddings.
    """
    print(f"Loading SBERT model: {model_name}...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(model_name, device=device)
    print("Encoding sentences...")
    embeddings = model.encode(texts.tolist(), show_progress_bar=True, batch_size=32)
    return embeddings

def train_evaluate_baselines(data_dir='data/splits', output_dir='results/baselines'):
    """
    Trains and evaluates baselines with DQE Augmentation:
    1. SBERT + SVM
    2. BOW + XGBoost
    3. TF-IDF + XGBoost
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # --- 1. Load Data ---
    print("Loading splits...")
    train_df = pd.read_csv(os.path.join(data_dir, 'train_full.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    
    # Handle NaN
    train_df['cleaned_text'] = train_df['cleaned_text'].fillna('')
    test_df['cleaned_text'] = test_df['cleaned_text'].fillna('')

    # --- 2. DQE Augmentation (Simulated via Semantic/Random Oversampling) ---
    print("\n--- Applying DQE Augmentation ---")
    dqe = DQEAugmenter(target_count='auto', random_state=42)
    train_df_aug = dqe.fit_resample(train_df, 'cleaned_text', 'cyberbullying_type')
    
    # Prepare Data
    X_train_text = train_df_aug['cleaned_text']
    y_train = train_df_aug['cyberbullying_type']
    X_test_text = test_df['cleaned_text']
    y_test = test_df['cyberbullying_type']
    
    # Encode Labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    
    # Save Mapping
    label_map = dict(zip(le.classes_, le.transform(le.classes_)))
    with open(os.path.join(output_dir, 'label_map.json'), 'w') as f:
        json.dump({str(k): int(v) for k, v in label_map.items()}, f, indent=2)

    # --- 3. Define Experiments ---
    experiments = [
        {
            'name': 'BOW_XGBoost',
            'vec_type': 'bow',
            'model': xgb.XGBClassifier(objective='multi:softprob', num_class=len(le.classes_), random_state=42, n_jobs=-1)
        },
        {
            'name': 'TFIDF_XGBoost',
            'vec_type': 'tfidf',
            'model': xgb.XGBClassifier(objective='multi:softprob', num_class=len(le.classes_), random_state=42, n_jobs=-1)
        },
        {
            'name': 'SBERT_SVM',
            'vec_type': 'sbert',
            # SVM doesn't output probs by default, use CalibratedClassifierCV
            'model': CalibratedClassifierCV(LinearSVC(class_weight='balanced', random_state=42, dual='auto')) 
        }
    ]
    
    results = []
    
    for exp in experiments:
        name = exp['name']
        print(f"\n--- Running Experiment: {name} ---")
        
        # Feature Extraction
        if exp['vec_type'] == 'bow':
            vectorizer = CountVectorizer(max_features=10000, ngram_range=(1, 1))
            X_train_feat = vectorizer.fit_transform(X_train_text)
            X_test_feat = vectorizer.transform(X_test_text)
            
        elif exp['vec_type'] == 'tfidf':
            vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
            X_train_feat = vectorizer.fit_transform(X_train_text)
            X_test_feat = vectorizer.transform(X_test_text)
            
        elif exp['vec_type'] == 'sbert':
            X_train_feat = get_embeddings(X_train_text)
            X_test_feat = get_embeddings(X_test_text)
            
        # Training
        model = exp['model']
        print(f"Training {name} on {X_train_feat.shape[0]} samples...")
        model.fit(X_train_feat, y_train_enc)
        
        # Evaluation
        print(f"Evaluating {name}...")
        y_pred = model.predict(X_test_feat)
        y_prob = model.predict_proba(X_test_feat)
        
        # Metrics
        macro_f1 = f1_score(y_test_enc, y_pred, average='macro')
        mcc = matthews_corrcoef(y_test_enc, y_pred)
        
        try:
            auprc = average_precision_score(pd.get_dummies(y_test_enc), y_prob, average='macro')
        except Exception:
            auprc = 0.0
            
        print(f"  Macro-F1: {macro_f1:.4f} | MCC: {mcc:.4f}")
        
        results.append({
            'Model': name,
            'Macro_F1': macro_f1,
            'MCC': mcc,
            'AUPRC': auprc
        })
        
        # Save Report
        report = classification_report(y_test_enc, y_pred, target_names=le.classes_, output_dict=True)
        with open(os.path.join(output_dir, f'{name}_report.json'), 'w') as f:
            json.dump(report, f, indent=2)

    # Save Summary
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'baseline_summary.csv'), index=False)
    print(f"\nBaselines Complete. Results saved to {output_dir}")

if __name__ == "__main__":
    train_evaluate_baselines()
