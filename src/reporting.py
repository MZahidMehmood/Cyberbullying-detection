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
            
            # Extract metadata from filename (e.g., results_neutral_0shot.csv or results_neutral_0shot_lowdata.csv)
            parts = filename.replace('.csv', '').split('_')
            strategy = parts[1] if len(parts) > 1 else 'unknown'
            shots = parts[2] if len(parts) > 2 else '0shot'
            is_lowdata = 'lowdata' in parts
            
            suffix = "_lowdata" if is_lowdata else ""
            
            # Confusion Matrix
            classes = df['cyberbullying_type'].unique()
            plot_confusion_matrix(
                df['cyberbullying_type'], 
                df['pred_label'], 
                classes, 
                os.path.join(output_dir, f'cm_{strategy}_{shots}{suffix}.png')
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
                
            # Error Analysis
            from src.evaluation import analyze_errors
            # Ensure 'cleaned_text' is present, or use 'text'
            text_col = 'cleaned_text' if 'cleaned_text' in df.columns else 'text'
            if text_col in df.columns:
                error_stats = analyze_errors(df, 'cyberbullying_type', 'pred_label', text_col)
                with open(os.path.join(output_dir, f'error_analysis_{strategy}_{shots}.json'), 'w') as f:
                    json.dump(error_stats, f, indent=2)
            
            # Collect stats for Pareto
            from sklearn.metrics import f1_score
            macro_f1 = f1_score(df['cyberbullying_type'], df['pred_label'], average='macro')
            avg_latency = df['latency'].mean() if 'latency' in df.columns else 0
            avg_vram = df['vram_mb'].mean() if 'vram_mb' in df.columns else 0
            avg_cost = df['cost_est'].mean() if 'cost_est' in df.columns else 0
            
            summary_data.append({
                'Model': 'LLM', # Placeholder, ideally passed in
                'Strategy': strategy,
                'Shots': shots,
                'Regime': 'Low Data' if is_lowdata else 'Full Data',
                'Macro_F1': macro_f1,
                'Latency': avg_latency,
                'VRAM': avg_vram,
                'Cost': avg_cost
            })
            
    # Pareto Plots
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        
        # 1. Latency
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=summary_df, x='Latency', y='Macro_F1', hue='Strategy', style='Regime', s=100)
        plt.title('Pareto Frontier: Performance vs Latency')
        plt.xlabel('Latency (s/sample)')
        plt.ylabel('Macro-F1 Score')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'pareto_frontier_latency.png'))
        plt.close()
        
        # 2. VRAM
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=summary_df, x='VRAM', y='Macro_F1', hue='Strategy', style='Regime', s=100)
        plt.title('Pareto Frontier: Performance vs VRAM')
        plt.xlabel('VRAM Usage (MB)')
        plt.ylabel('Macro-F1 Score')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'pareto_frontier_vram.png'))
        plt.close()
        
        # 3. Cost
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=summary_df, x='Cost', y='Macro_F1', hue='Strategy', style='Regime', s=100)
        plt.title('Pareto Frontier: Performance vs Cost')
        plt.xlabel('Estimated Cost ($)')
        plt.ylabel('Macro-F1 Score')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'pareto_frontier_cost.png'))
        plt.close()
        
        summary_df.to_csv(os.path.join(output_dir, 'summary_metrics.csv'), index=False)
        
        # Generate Markdown Table
        markdown_table = summary_df.to_markdown(index=False)
        with open(os.path.join(output_dir, 'summary_metrics.md'), 'w') as f:
            f.write("# Experiment Summary\n\n")
            f.write(markdown_table)
        
        # --- Post-Hoc Analysis: McNemar Test ---
        # Compare Top 2 models by Macro-F1
        if len(summary_df) >= 2:
            top_2 = summary_df.sort_values('Macro_F1', ascending=False).head(2)
            
            # Helper to find result file for a given row
            def find_result_file(row):
                # Try standard naming conventions
                # Expected: results_{strategy}_{shots}shot{_lowdata}.csv
                strategy = row['Strategy']
                shots = row['Shots']
                regime = row.get('Regime', 'Full Data')
                suffix = "_lowdata" if regime == "Low Data" else ""
                
                filename = f"results_{strategy}_{shots}shot{suffix}.csv"
                filepath = os.path.join(results_dir, filename)
                
                if os.path.exists(filepath):
                    return filepath
                return None

            file1 = find_result_file(top_2.iloc[0])
            file2 = find_result_file(top_2.iloc[1])
            
            if file1 and file2:
                print(f"Performing McNemar Test between:\n 1) {os.path.basename(file1)}\n 2) {os.path.basename(file2)}")
                df1 = pd.read_csv(file1)
                df2 = pd.read_csv(file2)
                
                # Align data (assuming same test set order)
                if len(df1) == len(df2):
                    from src.evaluation import perform_mcnemar
                    stat, pval = perform_mcnemar(df1['cyberbullying_type'], df1['pred_label'], df2['pred_label'])
                    
                    with open(os.path.join(output_dir, 'mcnemar_top2.txt'), 'w') as f:
                        f.write(f"Comparing Best ({os.path.basename(file1)}) vs Runner-up ({os.path.basename(file2)})\n")
                        f.write(f"Statistic: {stat}\n")
                        f.write(f"P-value: {pval}\n")
                        f.write(f"Significant (p<0.05): {pval < 0.05}\n")
                else:
                    print("Warning: Prediction files have different lengths, skipping McNemar.")
            else:
                print("Warning: Could not find result files for top 2 models, skipping McNemar.")


if __name__ == "__main__":
    # Define paths
    results_dir = os.path.join("results", "llm_experiments")
    output_dir = os.path.join("results", "reporting_artifacts")
    
    print(f"Generating artifacts from {results_dir} to {output_dir}...")
    
    if os.path.exists(results_dir):
        generate_artifacts(results_dir, output_dir)
        print("Artifact generation complete.")
    else:
        print(f"Results directory {results_dir} not found. Run experiments first.")
