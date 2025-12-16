import os
import sys
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType
import pandas as pd

def main():
    print("Starting LoRA Ablation Study...")
    
    model_name = "meta-llama/Meta-Llama-3-8B" # Example, user can change
    output_dir = r"H:\The Thesis\results\lora_ablation"
    
    # Load Data
    data_file = r"H:\The Thesis\data\splits\train_full.csv"
    if not os.path.exists(data_file):
        print("Data not found.")
        return
        
    df = pd.read_csv(data_file)
    # Convert to HuggingFace Dataset
    # Format: Instruction + Input -> Output
    # We need to format the data for SFT (Supervised Fine-Tuning)
    
    def format_example(row):
        return f"Classify this tweet: {row['cleaned_text']}\nLabel: {row['cyberbullying_type']}"
        
    df['text'] = df.apply(format_example, axis=1)
    dataset = Dataset.from_pandas(df[['text']])
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize(batch):
        return tokenizer(batch['text'], padding="max_length", truncation=True, max_length=128)
        
    tokenized_ds = dataset.map(tokenize, batched=True)
    
    # Model & LoRA Config
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False, 
        r=8, 
        lora_alpha=32, 
        lora_dropout=0.05
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Training Args
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        num_train_epochs=1,
        learning_rate=2e-4,
        logging_steps=10,
        save_steps=100,
        fp16=True if torch.cuda.is_available() else False
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8)
    )
    
    print("Training...")
    trainer.train()
    
    print(f"LoRA model saved to {output_dir}")
    model.save_pretrained(output_dir)
    
    # --- Evaluation (as per flowchart: F1/ECE/Abstention) ---
    print("Running Evaluation...")
    model.eval()
    
    # Load Test Data
    test_file = r"H:\The Thesis\data\splits\test.csv"
    if os.path.exists(test_file):
        test_df = pd.read_csv(test_file)
        # Limit for speed if needed, but flowchart implies full eval
        # test_df = test_df.head(100) 
        
        predictions = []
        labels = []
        confidences = []
        
        from tqdm import tqdm
        import numpy as np
        from sklearn.metrics import f1_score
        
        # Simple generation loop
        for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
            prompt = f"Classify this tweet: {row['cleaned_text']}\nLabel:"
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id)
            
            gen_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            
            # Naive parsing (expecting exact label match)
            pred_label = "unknown"
            for label in ['not_cyberbullying', 'gender', 'religion', 'other_cyberbullying', 'age', 'ethnicity']:
                if label in gen_text:
                    pred_label = label
                    break
            
            predictions.append(pred_label)
            labels.append(row['cyberbullying_type'])
            confidences.append(1.0) # Placeholder as generation doesn't give prob easily without logits analysis
            
        # Calculate Metrics
        macro_f1 = f1_score(labels, predictions, average='macro')
        print(f"LoRA Macro-F1: {macro_f1:.4f}")
        
        # Save results
        res_df = pd.DataFrame({'y_true': labels, 'y_pred': predictions, 'confidence': confidences})
        res_df.to_csv(os.path.join(output_dir, "lora_predictions.csv"), index=False)
        
        # Note: ECE and Abstention require probabilities. 
        # Since we are doing generation, we use confidence=1.0 proxy or would need logit access.
        # For strict flowchart compliance, we save the CSV so the 'reporting.py' module 
        # (or a custom script) can generate the plots.
    else:
        print("Test data not found, skipping evaluation.")

if __name__ == "__main__":
    main()
