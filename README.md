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

The easiest way to run the project is via the master runner:

```bash
python run_project.py
```

This will give you an interactive menu:
1.  **Preprocessing**: Cleans `cyberbullying_tweets.csv` and creates splits in `data/`.
2.  **Classical Baselines**: Trains LogReg, SVM, and XGBoost.
3.  **LLM Dry Run**: Quick test to verify setup.
4.  **FULL LLM SUITE**: Runs all models (Qwen, LLaMA, etc.) with Neutral vs. Aggressive strategies.
5.  **Evaluation**: Generates plots and tables in `results/`.

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
