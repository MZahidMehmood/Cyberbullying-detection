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
            idx1, idx2 = top_2.index[0], top_2.index[1]
            
            # We need to reload the dataframes to get predictions
            # This is a bit inefficient but robust
            # Reconstruct filenames from metadata (assuming standard naming)
            # Or better, store filepaths in summary_data
            # For this implementation, we will skip the complex reload and just print a placeholder
            # unless we stored filepaths. Let's assume we didn't store filepaths.
            # To do this right, we should have stored filepaths.
            pass 
            
            # Actually, let's do it properly.
            # We need to modify the loop above to store 'filepath' in summary_data.
            # But since we are in a 'replace_content' block, we can't easily change the loop structure 
            # without replacing the whole function.
            # Instead, we will rely on the user running 'run_all_experiments.py' which could do this.
            # However, reporting.py is the place for it.
            
            # Let's just add a comment that McNemar requires loading the specific predictions.
            # Given the constraints, we will implement Error Analysis inside the loop (easier)
            # and skip automated McNemar here to avoid breaking the file structure, 
            # or we can try to find the files again.
            
            # Let's try to find the files again based on strategy/shots.
            file1 = f"results_{top_2.iloc[0]['Strategy']}_{top_2.iloc[0]['Shots']}shot.csv"
            file2 = f"results_{top_2.iloc[1]['Strategy']}_{top_2.iloc[1]['Shots']}shot.csv"
            
            path1 = os.path.join(results_dir, file1)
            path2 = os.path.join(results_dir, file2)
            
            if os.path.exists(path1) and os.path.exists(path2):
                df1 = pd.read_csv(path1)
                df2 = pd.read_csv(path2)
                
                # Align data (assuming same test set order)
                if len(df1) == len(df2):
                    from src.evaluation import perform_mcnemar
                    stat, pval = perform_mcnemar(df1['cyberbullying_type'], df1['pred_label'], df2['pred_label'])
                    
                    with open(os.path.join(output_dir, 'mcnemar_top2.txt'), 'w') as f:
                        f.write(f"Comparing Best ({file1}) vs Runner-up ({file2})\n")
                        f.write(f"Statistic: {stat}\n")
                        f.write(f"P-value: {pval}\n")
                        f.write(f"Significant (p<0.05): {pval < 0.05}\n")


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
