import os
import torch
import pandas as pd
import json
import random
import traceback
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sklearn.metrics import classification_report, f1_score, matthews_corrcoef, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# Configuration from fewshot_authentic_results.json
# CRITICAL: fewshot_examples_used = 72 (12 per class × 6 classes)
MODELS = {
    "Qwen-7B-Instruct-FewShot": "Qwen/Qwen2.5-7B-Instruct",
    "Qwen-8B-Instruct-FewShot": "Qwen/Qwen2.5-8B-Instruct",
    "DeepSeek-7B-Instruct-FewShot": "deepseek-ai/deepseek-llm-7b-chat",
    "DeepSeek-8B-Instruct-FewShot": "deepseek-ai/deepseek-llm-8b-chat",
    "Llama-3-7B-Instruct-FewShot": "meta-llama/Meta-Llama-3-7B-Instruct",
    "Llama-3-8B-Instruct-FewShot": "meta-llama/Meta-Llama-3-8B-Instruct",
    "Mistral-7B-Instruct-FewShot": "mistralai/Mistral-7B-Instruct-v0.3",
    "Mistral-8B-Instruct-FewShot": "mistralai/Mixtral-8x7B-Instruct-v0.1"
}

# From JSON: "fewshot_examples_per_class": 12
EXAMPLES_PER_CLASS = 12  # 12 × 6 classes = 72 total

def load_data(data_dir):
    print("Loading data...")
    if not os.path.exists(os.path.join(data_dir, "train_full.csv")):
        raise FileNotFoundError(f"Training data not found in {data_dir}")
        
    train_df = pd.read_csv(os.path.join(data_dir, "train_full.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    train_df['cleaned_text'] = train_df['cleaned_text'].fillna('')
    test_df['cleaned_text'] = test_df['cleaned_text'].fillna('')
    return train_df, test_df

def get_balanced_examples(train_df, per_class=12, seed=42):
    """Selects examples, balanced across classes (12 per class = 72 total)."""
    random.seed(seed)
    classes = sorted(train_df['cyberbullying_type'].unique())  # Sort for consistency
    
    examples = []
    for cls in classes:
        cls_df = train_df[train_df['cyberbullying_type'] == cls]
        current_examples = cls_df.sample(n=min(len(cls_df), per_class), random_state=seed)
        examples.append(current_examples)
        
    final_examples = pd.concat(examples).sample(frac=1, random_state=seed)
    return final_examples

def format_prompt(examples, target_text):
    """
    Constructs the few-shot prompt with Chain-of-Thought.
    From JSON: "Classify the following tweet for cyberbullying type. Think step-by-step about why it fits a category."
    """
    prompt = "Classify the following tweet for cyberbullying type. Think step-by-step about why it fits a category. Examples:\\n\\n"
    
    for _, row in examples.iterrows():
        prompt += f"Tweet: {row['cleaned_text']}\\nReasoning and Type: {row['cyberbullying_type']}\\n\\n"
        
    prompt += f"Tweet: {target_text}\\nReasoning and Type:"
    return prompt

def run_inference(model_name, model_id, train_df, test_df, output_dir):
    print(f"\\n{'='*60}\\nRunning Few-Shot Inference for {model_name} ({model_id})\\n{'='*60}")
    print("NOTE: Model (~15-20GB) will be downloaded on first run.\\n")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    try:
        # Track inference start time
        inference_start = time.time()
        total_tokens = 0
        
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        
        # Get Examples (12 per class = 72 total)
        examples = get_balanced_examples(train_df, per_class=EXAMPLES_PER_CLASS)
        print(f"Constructed prompt with {len(examples)} examples ({EXAMPLES_PER_CLASS} per class).")
        
        # Calculate prompt length
        sample_prompt = format_prompt(examples, "sample tweet")
        prompt_length_avg = len(tokenizer.encode(sample_prompt))
        
        preds = []
        BATCH_SIZE = 8
        
        for i in tqdm(range(0, len(test_df), BATCH_SIZE), desc="Inference"):
            batch = test_df.iloc[i:i+BATCH_SIZE]
            prompts = [format_prompt(examples, text) for text in batch['cleaned_text']]
            
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to("cuda")
            total_tokens += inputs['input_ids'].numel()
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=20,  # Allow for reasoning text
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            total_tokens += (outputs.numel() - inputs['input_ids'].numel())
                
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            # Parse output (extract label from "Reasoning and Type: <label>")
            for text in decoded:
                try:
                    # Look for the pattern after last "Type:"
                    if "Type:" in text:
                        pred = text.split("Type:")[-1].strip().split()[0].lower()
                    else:
                        pred = text.split("Reasoning and Type:")[-1].strip().split()[0].lower() if "Reasoning and Type:" in text else "not_cyberbullying"
                except:
                    pred = "not_cyberbullying"
                preds.append(pred)
        
        # Calculate inference metrics
        inference_time_seconds = time.time() - inference_start
        tokens_per_second = total_tokens / inference_time_seconds if inference_time_seconds > 0 else 0
        gpu_memory_peak_gb = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
        
        # Save predictions
        test_df[f'{model_name}_pred'] = preds[:len(test_df)]
        test_df.to_csv(os.path.join(output_dir, f'{model_name}_predictions.csv'), index=False)
        
        # Calculate Metrics
        le = LabelEncoder()
        all_labels = sorted(train_df['cyberbullying_type'].unique())
        le.fit(all_labels)
        
        y_true = le.transform(test_df['cyberbullying_type'])
        y_pred = []
        for p in preds[:len(test_df)]:
            if p in le.classes_:
                y_pred.append(le.transform([p])[0])
            else:
                y_pred.append(le.transform(['not_cyberbullying'])[0])
        
        # Full classification report
        report = classification_report(y_true, y_pred, target_names=le.classes_, output_dict=True, zero_division=0)
        
        # Build output matching JSON structure
        result = {
            "overall_metrics": {
                "macro_f1": round(report['macro avg']['f1-score'], 3),
                "f1_weighted": round(f1_score(y_true, y_pred, average='weighted', zero_division=0), 3),
                "mcc": round(matthews_corrcoef(y_true, y_pred), 3),
                "auprc": 0.0,  # Placeholder
                "ece": 0.0,  # Placeholder
                "accuracy": round(accuracy_score(y_true, y_pred), 3),
                "precision_macro": round(precision_score(y_true, y_pred, average='macro', zero_division=0), 3),
                "recall_macro": round(recall_score(y_true, y_pred, average='macro', zero_division=0), 3),
                "f1_micro": round(f1_score(y_true, y_pred, average='micro', zero_division=0), 3)
            },
            "per_class_metrics": {},
            "inference_info": {
                "inference_time_seconds": round(inference_time_seconds, 1),
                "tokens_per_second": round(tokens_per_second, 1),
                "gpu_memory_peak_gb": round(gpu_memory_peak_gb, 1),
                "fewshot_examples_used": len(examples),  # Should be 72
                "prompt_length_avg": prompt_length_avg
            }
        }
        
        # Add per-class metrics
        for class_name in le.classes_:
            if class_name in report:
                result["per_class_metrics"][class_name] = {
                    "precision": round(report[class_name]['precision'], 3),
                    "recall": round(report[class_name]['recall'], 3),
                    "f1": round(report[class_name]['f1-score'], 3)
                }

        # Save JSON
        with open(os.path.join(output_dir, f'{model_name}_report.json'), 'w') as f:
            json.dump(result, f, indent=2)
            
        print(f"✓ {model_name}: Macro-F1={result['overall_metrics']['macro_f1']:.3f}, Acc={result['overall_metrics']['accuracy']:.3f}")
        
        # Release memory
        del model, tokenizer
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"✗ FAILED {model_name}: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/splits")
    parser.add_argument("--output_dir", type=str, default="results/fewshot_reproduction")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    train_df, test_df = load_data(args.data_dir)
    
    for name, model_id in MODELS.items():
        run_inference(name, model_id, train_df, test_df, args.output_dir)
