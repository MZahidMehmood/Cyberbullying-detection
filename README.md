# Cyberbullying Detection with Aggression-Enhanced Prompting

This repository contains the implementation for the MS Thesis project: **"Cyberbullying Detection via Aggression-Enhanced Prompting"**.

The project reproduces results for Large Language Models (Qwen, DeepSeek, Llama-3, Mistral) using two key strategies:
1.  **Supervised Fine-Tuning (SFT)**: Using LoRA adapters for efficient fine-tuning.
2.  **Advanced Few-Shot Inference**: Using 72-shot Chain-of-Thought (CoT) prompting.

> [!NOTE]
> **Status**: ✅ Fully Verified & Audited (Matches Authentic Results)

## Project Structure

```text
.
├── src/
│   ├── reproduce_sft.py      # Main SFT Pipeline (LoRA Training)
│   ├── reproduce_fewshot.py  # Main Few-Shot Pipeline (72-shot CoT)
│   ├── preprocessing.py      # Data cleaning and normalization
│   ├── create_splits.py      # Stratified splitting
│   ├── evaluation.py         # Metrics calculation
│   └── reporting.py          # Reporting artifacts
├── run_project.py            # Master orchestrator script
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/MZahidMehmood/Cyberbullying-detection.git
    cd Cyberbullying-detection
    ```
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Flash Attention 2 is recommended for faster training if supported by your GPU.*

## Usage

### 1. Reproduce Full Project (Recommended)
Run the complete reproduction pipeline (SFT + Few-Shot):

```bash
python run_project.py
```

This script will:
1.  Verify/Create data splits.
2.  Run SFT for all 8 models (Qwen, DeepSeek, Llama-3, Mistral - 7B/8B).
3.  Run Few-Shot inference for all 8 models.
4.  Generate comparison reports against authentic results.

### 2. Run Individual Modules

**Supervised Fine-Tuning (SFT):**
```bash
python src/reproduce_sft.py
```
*   Trains 8 models using LoRA (`rank=16`, `alpha=32`, `lr=1.5e-5`).
*   Outputs JSON reports to `results/sft_reproduction/`.

**Few-Shot Inference:**
```bash
python src/reproduce_fewshot.py
```
*   Runs inference using 72 examples (12 per class) with Chain-of-Thought.
*   Outputs JSON reports to `results/fewshot_reproduction/`.

## Methodology

*   **Dataset**: Cyberbullying Tweets (~47k samples).
*   **Models**: Qwen-2.5, DeepSeek-LLM, Llama-3, Mistral/Mixtral (7B & 8B variants).
*   **SFT Approach**: 4-bit Quantization + LoRA Fine-tuning.
*   **Few-Shot Approach**: "Diverse Balanced Sampling" (12 examples/class) + CoT Prompting.
*   **Metrics**: Macro-F1, Weighted-F1, MCC, Accuracy.

## Results

Reference results are stored in:
*   `results/sft_reproduction/detailed_results.json`
*   `results/fewshot_reproduction/fewshot_results.json`

The reproduction scripts generate verification reports to confirm alignment with these authentic results.
