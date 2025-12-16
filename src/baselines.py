import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, f1_score, matthews_corrcoef, average_precision_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import os
import json

def train_evaluate_baselines(data_dir='data/splits', output_dir='results/baselines'):
    """
    Trains and evaluates classical baselines:
    1. TF-IDF Vectorization
    2. Models: LogReg, SVM (calibrated), XGBoost
    3. Metrics: Macro-F1, MCC, AUPRC
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Data
    print("Loading data...")
    train_df = pd.read_csv(os.path.join(data_dir, 'train_full.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    
    # Handle NaN in text
    train_df['cleaned_text'] = train_df['cleaned_text'].fillna('')
    test_df['cleaned_text'] = test_df['cleaned_text'].fillna('')
    
    X_train_text = train_df['cleaned_text']
    y_train = train_df['cyberbullying_type']
    X_test_text = test_df['cleaned_text']
    y_test = test_df['cyberbullying_type']
    
    # Encode Labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    
    # Save Label Mapping
    label_map = dict(zip(le.classes_, le.transform(le.classes_)))
    with open(os.path.join(output_dir, 'label_map.json'), 'w') as f:
        json.dump({str(k): int(v) for k, v in label_map.items()}, f, indent=2)
        
    # TF-IDF Vectorization
    print("Vectorizing text...")
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train_vec = tfidf.fit_transform(X_train_text)
    X_test_vec = tfidf.transform(X_test_text)
    
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        'SVM': CalibratedClassifierCV(LinearSVC(class_weight='balanced', random_state=42, dual='auto')),
        'XGBoost': xgb.XGBClassifier(objective='multi:softprob', num_class=len(le.classes_), random_state=42)
    }
    
    results = []
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_vec, y_train_enc)
        
        print(f"Evaluating {name}...")
        y_pred = model.predict(X_test_vec)
        y_prob = model.predict_proba(X_test_vec)
        
        # Metrics
        macro_f1 = f1_score(y_test_enc, y_pred, average='macro')
        mcc = matthews_corrcoef(y_test_enc, y_pred)
        
        # AUPRC (One-vs-Rest average)
        auprc = 0
        try:
            # For multi-class, we calculate AUPRC per class and average
            auprc = average_precision_score(pd.get_dummies(y_test_enc), y_prob, average='macro')
        except Exception as e:
            print(f"Warning: AUPRC calculation failed for {name}: {e}")
            
        print(f"  Macro-F1: {macro_f1:.4f}")
        
        results.append({
            'Model': name,
            'Macro_F1': macro_f1,
            'MCC': mcc,
            'AUPRC': auprc
        })
        
        # Save Classification Report
        report = classification_report(y_test_enc, y_pred, target_names=le.classes_, output_dict=True)
        with open(os.path.join(output_dir, f'{name}_report.json'), 'w') as f:
            json.dump(report, f, indent=2)
            
    # Save Summary Results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'baseline_summary.csv'), index=False)
    print(f"Baseline evaluation complete. Results saved to {output_dir}")
