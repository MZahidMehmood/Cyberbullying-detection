import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import json

def plot_confusion_matrix(y_true, y_pred, classes, output_path):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_calibration_curve(y_true, y_prob, output_path, n_bins=10):
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    
    plt.figure(figsize=(8, 8))
    plt.plot(prob_pred, prob_true, marker='o', label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')
    plt.xlabel('Mean Predicted Confidence')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_coverage_risk(y_true, y_pred, confidence, output_path):
    """
    Plots Coverage-Risk curve: Risk (Error Rate) vs Coverage (Fraction of samples retained).
    """
    import numpy as np
    
    # Sort by confidence descending
    data = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'conf': confidence})
    data = data.sort_values('conf', ascending=False)
    
    coverages = []
    risks = []
    
    n = len(data)
    for i in range(1, n + 1):
        subset = data.iloc[:i]
        coverage = i / n
        accuracy = (subset['y_true'] == subset['y_pred']).mean()
        risk = 1.0 - accuracy
        
        coverages.append(coverage)
        risks.append(risk)
        
    plt.figure(figsize=(8, 6))
    plt.plot(coverages, risks, label='Model Risk')
    plt.xlabel('Coverage (Fraction of samples retained)')
    plt.ylabel('Risk (Error Rate)')
    plt.title('Coverage-Risk Curve (Abstention)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_pareto_frontier(metrics_df, output_path):
    """
    Plots Pareto frontier: Macro-F1 vs Latency (or Cost/VRAM).
    metrics_df should have columns: ['Model', 'Macro_F1', 'Latency']
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=metrics_df, x='Latency', y='Macro_F1', hue='Model', style='Strategy', s=100)
    
    # Optional: Draw frontier line if needed, but scatter is usually sufficient for comparison
    
    plt.title('Pareto Frontier: Performance vs Latency')
    plt.xlabel('Latency (s/sample)')
    plt.ylabel('Macro-F1 Score')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def generate_artifacts(results_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Aggregate results for Pareto
    summary_data = []
    
    for filename in os.listdir(results_dir):
        if filename.endswith('.csv') and 'results' in filename:
            filepath = os.path.join(results_dir, filename)
            df = pd.read_csv(filepath)
            
            # Extract metadata from filename (e.g., results_neutral_0shot.csv)
            parts = filename.replace('.csv', '').split('_')
            strategy = parts[1] if len(parts) > 1 else 'unknown'
            shots = parts[2] if len(parts) > 2 else '0shot'
            
            # Confusion Matrix
            classes = df['cyberbullying_type'].unique()
            plot_confusion_matrix(
                df['cyberbullying_type'], 
                df['pred_label'], 
                classes, 
                os.path.join(output_dir, f'cm_{strategy}_{shots}.png')
            )
            
            # Calibration (using binary correctness as proxy for multi-class confidence calibration)
            if 'confidence' in df.columns:
                is_correct = (df['cyberbullying_type'] == df['pred_label']).astype(int)
                plot_calibration_curve(
                    is_correct, 
                    df['confidence'], 
                    os.path.join(output_dir, f'calibration_{strategy}_{shots}.png')
                )
                
                # Abstention: Coverage-Risk
                plot_coverage_risk(
                    df['cyberbullying_type'],
                    df['pred_label'],
                    df['confidence'],
                    os.path.join(output_dir, f'coverage_risk_{strategy}_{shots}.png')
                )
            
            # Collect stats for Pareto
            from sklearn.metrics import f1_score
            macro_f1 = f1_score(df['cyberbullying_type'], df['pred_label'], average='macro')
            avg_latency = df['latency'].mean() if 'latency' in df.columns else 0
            
            summary_data.append({
                'Model': 'LLM', # Placeholder, ideally passed in
                'Strategy': strategy,
                'Shots': shots,
                'Macro_F1': macro_f1,
                'Latency': avg_latency
            })
            
    # Pareto Plot
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        plot_pareto_frontier(summary_df, os.path.join(output_dir, 'pareto_frontier.png'))
        summary_df.to_csv(os.path.join(output_dir, 'summary_metrics.csv'), index=False)
