import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import time
import random
import pandas as pd
from typing import List, Dict, Optional
import gc

class LLMPipeline:
    def __init__(self, model_name: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model_name = model_name
        self.device = device
        print(f"Loading model: {model_name} on {device}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True
            )
            self.model.eval()
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            raise e

    def extract_aggression_cues(self, train_df: pd.DataFrame, top_n: int = 20) -> List[str]:
        """
        Extracts top aggression keywords from the training set using simple frequency 
        difference or TF-IDF (simplified here to top words in bullying classes).
        """
        # This is a placeholder for a more sophisticated extraction.
        # For now, we return a static list or simple extraction.
        # In a real run, we'd use the TF-IDF features from the baseline.
        bullying_text = " ".join(train_df[train_df['cyberbullying_type'] != 'not_cyberbullying']['cleaned_text'])
        # Simple split and count (naive) - replace with TF-IDF top words if available
        # For this implementation, we will use a predefined list of common cues for robustness
        # unless dynamic extraction is strictly required.
        cues = ["idiot", "stupid", "kill", "ugly", "fat", "hate", "dumb", "loser", "suck", "nasty"]
        return cues

    def construct_prompt(self, 
                         text: str, 
                         strategy: str = "neutral", 
                         shots: List[Dict] = [], 
                         cues: List[str] = []) -> str:
        
        instruction = """You are an expert cyberbullying detection system.
Classify the following tweet into one of these categories: 
[not_cyberbullying, gender, religion, other_cyberbullying, age, ethnicity].

Output MUST be a valid JSON object with these fields:
- "label": The predicted class.
- "confidence": A score between 0.0 and 1.0.
- "rationale": A brief explanation.
"""

        if strategy == "aggressive":
            instruction += f"\nNOTE: Pay special attention to aggressive keywords and hostility markers such as: {', '.join(cues)}."

        prompt = f"{instruction}\n\n"
        
        # Few-shot examples
        for shot in shots:
            prompt += f"Tweet: {shot['text']}\nOutput: {json.dumps({'label': shot['label'], 'confidence': 1.0, 'rationale': 'Example'})}\n\n"
            
        prompt += f"Tweet: {text}\nOutput:"
        return prompt

    def generate(self, prompt: str, max_new_tokens: int = 200) -> Dict:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens, 
                temperature=0.01, # Near-deterministic
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        latency = time.time() - start_time
        
        input_tokens = inputs.input_ids.shape[1]
        output_tokens = outputs[0].shape[0] - input_tokens
        
        generated_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return {
            "text": generated_text.strip(), 
            "latency": latency,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }

    def parse_output(self, output_text: str) -> Optional[Dict]:
        try:
            # Attempt to find JSON structure
            start = output_text.find('{')
            end = output_text.rfind('}') + 1
            if start == -1 or end == 0:
                return None
            
            json_str = output_text[start:end]
            data = json.loads(json_str)
            
            if "label" in data and "confidence" in data:
                return data
            return None
        except:
            return None

    def predict(self, text: str, strategy: str, shots: List[Dict], cues: List[str], retries: int = 1) -> Dict:
        prompt = self.construct_prompt(text, strategy, shots, cues)
        
        for attempt in range(retries + 1):
            result = self.generate(prompt)
            parsed = self.parse_output(result["text"])
            
            if parsed:
                parsed['latency'] = result['latency']
                return parsed
            
            # Retry logic: could append "Invalid JSON, try again" to prompt
            # For now, we just re-generate (stochasticity might help if temp > 0, but we use temp~0)
            # So strictly speaking, without changing prompt/temp, retry won't help much.
            # We will skip complex retry logic for this MVP.
            
        return {"label": "error", "confidence": 0.0, "rationale": "Failed to generate valid JSON", "latency": result['latency']}

    def run_batch(self, 
                  texts: List[str], 
                  strategy: str, 
                  train_df: pd.DataFrame, 
                  n_shots: int = 0) -> List[Dict]:
        
        cues = self.extract_aggression_cues(train_df) if strategy == "aggressive" else []
        results = []
        
        # Pre-sample shots if needed (stratified sampling)
        shots = []
        if n_shots > 0:
            # Simple random sample for now. In production, use stratified.
            sample = train_df.sample(n=n_shots)
            for _, row in sample.iterrows():
                shots.append({"text": row['cleaned_text'], "label": row['cyberbullying_type']})
        
        print(f"Running inference: Strategy={strategy}, Shots={n_shots}, Count={len(texts)}")
        
        for i, text in enumerate(texts):
            if i % 10 == 0:
                print(f"  Processed {i}/{len(texts)}")
                
            # Dynamic shots could be implemented here (e.g., RAG-based selection)
            # For now, fixed shots for the batch
            
            res = self.predict(text, strategy, shots, cues)
            results.append(res)
            
        return results

def unload_model(pipeline):
    del pipeline.model
    del pipeline.tokenizer
    torch.cuda.empty_cache()
    gc.collect()
