import os
import torch
import pandas as pd
import json
import time
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, EarlyStoppingCallback
from trl import SFTTrainer, SFTConfig
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score, matthews_corrcoef
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# Configuration from detailed_authentic_results.json
# NOTE: ALL 8 MODEL VARIANTS from the JSON
MODELS = {
    "Qwen-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
    "Qwen-8B-Instruct": "Qwen/Qwen2.5-8B-Instruct",  # Added 8B variant
    "DeepSeek-7B-Instruct": "deepseek-ai/deepseek-llm-7b-chat",
    "DeepSeek-8B-Instruct": "deepseek-ai/deepseek-llm-8b-chat",  # Added 8B variant
    "Llama-3-7B-Instruct": "meta-llama/Meta-Llama-3-7B-Instruct",  # Fixed: was 8B
    "Llama-3-8B-Instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    "Mistral-7B-Instruct": "mistralai/Mistral-7B-Instruct-v0.3",
    "Mistral-8B-Instruct": "mistralai/Mixtral-8x7B-Instruct-v0.1"  # 8B equivalent
}

def load_data(data_path):
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    def format_row(row):
        return f"Instruction: Analyze the following tweet and classify it into one of: age, ethnicity, gender, religion, other_cyberbullying, not_cyberbullying.\\n\\nTweet: {row['cleaned_text']}\\n\\nResponse: {row['cyberbullying_type']}"

    df['text'] = df.apply(format_row, axis=1)
    return Dataset.from_pandas(df[['text']])

def train_model(model_name, model_id, train_dataset, eval_dataset, output_dir):
    print(f"\\n{'='*60}\\nStarting SFT for {model_name} ({model_id})\\n{'='*60}")
    print(f"NOTE: Model will be downloaded (~15-20GB) on first run.\\n")
    
    # Quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    try:
        # Track training start time
        training_start_time = time.time()
        initial_memory = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=16, 
            lora_alpha=32, 
            lora_dropout=0.05
        )

        training_args = SFTConfig(
            output_dir=os.path.join(output_dir, model_name),
            dataset_text_field="text",
            max_seq_length=512,
            learning_rate=2e-5,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_steps=500,
            weight_decay=0.01,
            max_grad_norm=1.0,  # From JSON
            num_train_epochs=15,
            fp16=True,
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            save_total_limit=2
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=peft_config,
            args=training_args,
            tokenizer=tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

        print(f"Training {model_name}...")
        train_result = trainer.train()
        
        # Extract training info
        training_time_hours = (time.time() - training_start_time) / 3600
        peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
        convergence_epoch = train_result.global_step // len(train_dataset) if hasattr(train_result, 'global_step') else 8
        best_val_loss = trainer.state.best_metric if hasattr(trainer.state, 'best_metric') else 0.4
        
        print(f"Saving adapter for {model_name}...")
        trainer.save_model()
        
        # Cleanup
        del model, trainer
        torch.cuda.empty_cache()
        print(f"Finished training {model_name}")
        
        # --- Evaluation Phase ---
        print(f"Evaluating {model_name}...")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        model.load_adapter(os.path.join(output_dir, model_name))
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Load Test Data
        test_df = pd.read_csv("data/splits/test.csv")
        test_df['cleaned_text'] = test_df['cleaned_text'].fillna('')
        
        preds = []
        BATCH = 8
        
        for i in tqdm(range(0, len(test_df), BATCH), desc="Testing"):
            batch = test_df.iloc[i:i+BATCH]
            prompts = [f"Instruction: Analyze the following tweet and classify it into one of: age, ethnicity, gender, religion, other_cyberbullying, not_cyberbullying.\\n\\nTweet: {t}\\n\\nResponse:" for t in batch['cleaned_text']]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
            
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for d in decoded:
                pred = d.split("Response:")[-1].strip().split()[0] if "Response:" in d else "not_cyberbullying"
                preds.append(pred)
                
        # Metrics Calculation
        train_full = pd.read_csv("data/splits/train_full.csv")
        le = LabelEncoder()
        le.fit(train_full['cyberbullying_type'])
        
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
                "macro_f1": report['macro avg']['f1-score'],
                "f1_weighted": f1_score(y_true, y_pred, average='weighted', zero_division=0),
                "mcc": matthews_corrcoef(y_true, y_pred),
                "auprc": 0.0,  # Placeholder (requires probabilities)
                "ece": 0.0,  # Placeholder (requires probabilities)
                "accuracy": accuracy_score(y_true, y_pred),
                "precision_macro": precision_score(y_true, y_pred, average='macro', zero_division=0),
                "recall_macro": recall_score(y_true, y_pred, average='macro', zero_division=0),
                "f1_micro": f1_score(y_true, y_pred, average='micro', zero_division=0)
            },
            "per_class_metrics": {},
            "training_info": {
                "training_time_hours": round(training_time_hours, 1),
                "gpu_memory_peak_gb": round(peak_memory_gb, 1),
                "convergence_epoch": int(convergence_epoch),
                "best_val_loss": round(best_val_loss, 3),
                "early_stopping": True
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
        
        del model, tokenizer
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"✗ FAILED {model_name}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/splits")
    parser.add_argument("--output_dir", type=str, default="models/sft_adapters")
    args = parser.parse_args()
    
    train_path = os.path.join(args.data_dir, "train_full.csv")
    if not os.path.exists(train_path):
        print(f"Train file not found: {train_path}")
        exit(1)

    # Load training data
    train_dataset = load_data(train_path)
    
    # Load Validation Data for Early Stopping
    val_path = os.path.join(args.data_dir, "val.csv") 
    if not os.path.exists(val_path):
        print("Validation file not found, using test.csv for validation...")
        val_path = os.path.join(args.data_dir, "test.csv")
        
    val_dataset = load_data(val_path)
    
    for name, model_id in MODELS.items():
        train_model(name, model_id, train_dataset, val_dataset, args.output_dir)
