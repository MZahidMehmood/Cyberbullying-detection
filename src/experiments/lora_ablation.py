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
        lora_dropout=0.1
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

if __name__ == "__main__":
    main()
