# Prompt Pack
# This file contains the prompt templates used in the experiments.

NEUTRAL_INSTRUCTION = """You are an expert cyberbullying detection system.
Classify the following tweet into one of these categories: 
[not_cyberbullying, gender, religion, other_cyberbullying, age, ethnicity].

Output MUST be a valid JSON object with these fields:
- "label": The predicted class.
- "confidence": A score between 0.0 and 1.0.
- "rationale": A brief explanation.
"""

AGGRESSIVE_INSTRUCTION_SUFFIX = "\nNOTE: Pay special attention to aggressive keywords and hostility markers such as: {cues}."

def get_prompt(text, strategy="neutral", shots=[], cues=[]):
    instruction = NEUTRAL_INSTRUCTION
    
    if strategy == "aggressive":
        cues_str = ", ".join(cues)
        instruction += AGGRESSIVE_INSTRUCTION_SUFFIX.format(cues=cues_str)
        
    prompt = f"{instruction}\n\n"
    
    # Few-shot examples
    import json
    for shot in shots:
        example_output = json.dumps({'label': shot['label'], 'confidence': 1.0, 'rationale': 'Example'})
        prompt += f"Tweet: {shot['text']}\nOutput: {example_output}\n\n"
        
    prompt += f"Tweet: {text}\nOutput:"
    return prompt
