import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef, average_precision_score, confusion_matrix
from statsmodels.stats.contingency_tables import mcnemar
import os
import json

def calculate_ece(y_true, y_prob, n_bins=10):
    """
    Calculates Expected Calibration Error (ECE).
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Filter for samples in this bin
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = (y_true[in_bin] == 1).mean() # Assuming binary correctness for ECE
            avg_confidence_in_bin = y_prob[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return ece

def perform_mcnemar(y_true, pred1, pred2):
    """
    Performs McNemar's test to compare two classifiers.
    Returns the statistic and p-value.
    """
    # Contingency table
    #           Model 2 Correct  Model 2 Wrong
    # Model 1 Correct    a             b
    # Model 1 Wrong      c             d
    
    correct1 = (pred1 == y_true)
    correct2 = (pred2 == y_true)
    
    a = np.sum(correct1 & correct2)
    b = np.sum(correct1 & ~correct2)
    c = np.sum(~correct1 & correct2)
    d = np.sum(~correct1 & ~correct2)
    
    table = [[a, b], [c, d]]
    result = mcnemar(table, exact=True)
    return result.statistic, result.pvalue

def analyze_errors(df, y_true_col, y_pred_col, text_col):
    """
    Analyzes errors based on specific categories:
    - Sarcasm (keyword based)
    - Code-Mix (non-ascii detection)
    - Implicit (high confidence errors)
    """
    df['is_correct'] = df[y_true_col] == df[y_pred_col]
    errors = df[~df['is_correct']].copy()
    
    # Sarcasm (Simple keyword list for demonstration)
    sarcasm_keywords = ['lol', 'lmao', 'jk', 'sarcasm', 'right', 'sure']
    errors['is_sarcasm'] = errors[text_col].apply(lambda x: any(k in str(x).lower() for k in sarcasm_keywords))
    
    # Code-Mix (Detect non-ASCII characters)
    errors['is_codemix'] = errors[text_col].apply(lambda x: any(ord(c) > 127 for c in str(x)))
    
    # Implicit (High confidence but wrong)
    if 'confidence' in errors.columns:
        errors['is_implicit'] = errors['confidence'] > 0.9
    else:
        errors['is_implicit'] = False
        
    return {
        'total_errors': len(errors),
        'sarcasm_errors': errors['is_sarcasm'].sum(),
        'codemix_errors': errors['is_codemix'].sum(),
        'implicit_errors': errors['is_implicit'].sum(),
        'error_samples': errors.head(5).to_dict(orient='records')
    }

def evaluate_predictions(results_file, ground_truth_file, output_dir):
    """
    Main evaluation function.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Evaluating {results_file}...")
    df = pd.read_csv(results_file)
    
    # Ensure ground truth is aligned (assuming same order or merge on ID if available)
    # For now, assuming 1:1 mapping as per run_llm_experiments.py
    
    y_true = df['cyberbullying_type']
    y_pred = df['pred_label']
    
    # Metrics
    metrics = {}
    metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro')
    metrics['per_class_f1'] = f1_score(y_true, y_pred, average=None).tolist() # Convert to list for JSON serialization
    metrics['classes'] = sorted(list(set(y_true) | set(y_pred))) # Save class names for reference
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    
    # AUPRC (Simplified: One-vs-Rest using confidence as proxy)
    # Since we don't have full probabilities, we treat 'confidence' as the prob of the predicted class.
    # This is an approximation.
    try:
        # Create a binary target: 1 if correct, 0 if wrong
        # And use confidence as the score. This gives AUPRC for "Correctness Detection".
        # Alternatively, for multi-class, we'd need full probs. 
        # Here we report AUPRC of the "Correctness" classifier as a proxy for model reliability.
        is_correct = (y_true == y_pred).astype(int)
        metrics['auprc_correctness'] = average_precision_score(is_correct, df['confidence'])
    except:
        metrics['auprc_correctness'] = 0.0
    
    # ECE
    if 'confidence' in df.columns:
        # Create binary correctness target for ECE
        is_correct = (y_true == y_pred).astype(int)
        metrics['ece'] = calculate_ece(is_correct, df['confidence'])
        
    # Error Analysis
    error_stats = analyze_errors(df, 'cyberbullying_type', 'pred_label', 'cleaned_text')
    metrics['error_analysis'] = error_stats
    
    # Save metrics
    base_name = os.path.basename(results_file).replace('.csv', '')
    with open(os.path.join(output_dir, f'{base_name}_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
        
    print(f"Metrics saved to {output_dir}")
    return metrics
