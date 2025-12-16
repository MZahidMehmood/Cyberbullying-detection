# Cyberbullying Detection with Aggression-Enhanced Prompting

This repository contains the implementation for the MS Thesis project: **"Cyberbullying Detection via Aggression-Enhanced Prompting"**.

The project compares classical machine learning baselines (XGBoost, SVM) against Large Language Models (Qwen, DeepSeek, LLaMA, Mistral) using a novel prompting strategy that incorporates aggression cues.

## Project Structure

```text
.
├── src/
│   ├── baselines.py        # Classical models (TF-IDF + ML)
│   ├── llm_core.py         # Main LLM pipeline (Prompting logic)
│   ├── preprocessing.py    # Data cleaning and normalization
│   ├── data_manager.py     # Stratified splitting and k-fold
│   ├── evaluation.py       # Metrics (F1, MCC, ECE) and Error Analysis
│   ├── reporting.py        # Plotting (Pareto, Confusion Matrix)
│   └── experiments/        # Appendices (Gemma replication, LoRA)
├── run_project.py          # MASTER RUNNER script
├── run_all_experiments.py  # Batch runner for all LLM configs
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Installation

1.  **Clone the repository** (or copy files).
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: For GPU acceleration, ensure you have the correct PyTorch version for your CUDA driver.*

## Usage

### 1. One-Command Execution (Recommended)
Run the entire pipeline (Preprocessing -> Splits -> Baselines -> LLM Experiments -> Reporting -> Appendices) with a single command:

```bash
python run_project.py
```

This script will:
1.  Clean the raw data (`cyberbullying_tweets.csv`).
2.  Generate stratified train/test splits and 5-fold CV folds.
3.  Train and evaluate classical baselines (LogReg, SVM, XGBoost).
4.  Run the full suite of LLM experiments (5 Models x 2 Strategies x 3 Shot settings x 2 Data Regimes).
5.  Generate all reporting artifacts (Pareto plots, Confusion Matrices, Error Analysis).
6.  Run Appendix experiments (Gemma Replication, LoRA Ablation).

### 2. Manual Execution
You can also run individual stages:

**Preprocessing & Splits:**
```bash
python src/preprocessing.py
python src/create_splits.py
```

**Classical Baselines:**
```bash
python src/baselines.py
```

**LLM Experiments:**
```bash
python run_all_experiments.py
```

**Reporting & Analysis:**
```bash
python src/reporting.py
```

## Methodology

*   **Dataset**: SOSNet (Cyberbullying Tweets), ~47k samples.
*   **Baselines**: TF-IDF features with Logistic Regression, SVM, and XGBoost.
*   **LLM Approach**:
    *   **Neutral Prompt**: Standard classification instruction.
    *   **Aggression-Enhanced Prompt**: Injects specific hostility markers into the prompt context.
    *   **Few-Shot**: Evaluated at 0, 4, and 8 shots.
*   **Metrics**: Macro-F1, MCC, AUPRC, Expected Calibration Error (ECE).

## Requirements

*   Python 3.8+
*   GPU with 16GB+ VRAM recommended for 7B/8B models (or use 4-bit quantization).
*   OpenAI/HuggingFace API key (if using API-based models, though this repo defaults to local execution).
