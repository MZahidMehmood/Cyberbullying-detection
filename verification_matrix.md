# Project Verification Matrix

| Flowchart Node | Requirement | Implemented In | Status | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Data Source** | SOSNet (~47.7k) | `src/preprocessing.py` | ✅ Verified | Loads `cyberbullying_tweets.csv` |
| **Preprocessing** | NFKC Normalization | `src/preprocessing.py` | ✅ Verified | `unicodedata.normalize('NFKC', ...)` |
| | Deduplication | `src/preprocessing.py` | ✅ Verified | `df.drop_duplicates` |
| | Strip URLs/@/RT | `src/preprocessing.py` | ✅ Verified | Regex substitution |
| | Keep Hashtags | `src/preprocessing.py` | ✅ Verified | Regex preserves `#` |
| | Collapse Repeats | `src/preprocessing.py` | ✅ Verified | `re.sub(r'(.)\1{2,}', ...)` |
| | Lowercase | `src/preprocessing.py` | ✅ Verified | `text.lower()` |
| | Emoji to Text | `src/preprocessing.py` | ✅ Verified | `emoji.demojize` |
| **Splits** | 75/25 Stratified | `src/create_splits.py` | ✅ Verified | `train_test_split(..., stratify=y)` |
| | 5-Fold CV | `src/create_splits.py` | ✅ Verified | `StratifiedKFold(n_splits=5)` |
| | Fixed Seeds | `src/create_splits.py` | ✅ Verified | `random_state=42` |
| | Low-Data Regime | `run_all_experiments.py` | ✅ Verified | Loop over `data_regimes = [False, True]` |
| **Classical Branch** | TF-IDF | `src/baselines.py` | ✅ Verified | `TfidfVectorizer` |
| | Models (LogReg, SVM, XGB) | `src/baselines.py` | ✅ Verified | Implemented with CV |
| **LLM Branch** | Prompt Construction | `src/prompts.py` | ✅ Verified | `get_prompt` function |
| | Neutral Strategy | `src/prompts.py` | ✅ Verified | `NEUTRAL_INSTRUCTION` |
| | Aggressive Strategy | `src/prompts.py` | ✅ Verified | `AGGRESSIVE_INSTRUCTION_SUFFIX` |
| | Few-Shot (0, 4, 8) | `run_all_experiments.py` | ✅ Verified | `shots_list = [0, 4, 8]` |
| | Models (Qwen, DeepSeek, LLaMA, Mistral, Mixtral) | `run_all_experiments.py` | ✅ Verified | `models` list matches exactly |
| | Inference (Fixed Decoding) | `src/llm_core.py` | ✅ Verified | `temperature=0.01` (deterministic) |
| | Output Enforcement | `src/llm_core.py` | ✅ Verified | `parse_output` (JSON) |
| | 1 Retry | `src/llm_core.py` | ✅ Verified | `retries=1` loop |
| | Log Tokens/Latency/VRAM/$ | `src/llm_core.py` | ✅ Verified | Returns dict with all metrics |
| **Appendices** | Gemma Replication | `src/experiments/gemma_replication.py` | ✅ Verified | Reuses `LLMPipeline` (same prompts/decoding) |
| | LoRA Ablation | `src/experiments/lora_ablation.py` | ✅ Verified | `r=8`, `alpha=32`, `dropout=0.05` |
| **Evaluation** | Metrics (Macro-F1, MCC, AUPRC) | `src/evaluation.py` | ✅ Verified | `f1_score`, `matthews_corrcoef`, `average_precision_score` |
| | Per-Class F1 | `src/evaluation.py` | ✅ Verified | `classification_report` output |
| | Calibration (ECE) | `src/reporting.py` | ✅ Verified | `plot_calibration_curve` |
| | McNemar Test | `src/reporting.py` | ✅ Verified | `perform_mcnemar` with robust file finding |
| | Pareto Plots | `src/reporting.py` | ✅ Verified | Latency, VRAM, and Cost plots |
| | Abstention (Coverage-Risk) | `src/reporting.py` | ✅ Verified | `plot_coverage_risk` |
| | Error Analysis | `src/evaluation.py` | ✅ Verified | Detects Sarcasm (keywords), Code-mix (ASCII), Implicit (Conf) |
| **Artifacts** | Tables/Plots | `src/reporting.py` | ✅ Verified | Saves .png and .csv |
| | Prompts | `run_llm_experiments.py` | ✅ Verified | Saves `_prompt.txt` |
| | Model Cards | `run_llm_experiments.py` | ✅ Verified | Saves `_model_card.md` |
| | Checksums | `run_llm_experiments.py` | ✅ Verified | Saves `_checksum.sha256` |
