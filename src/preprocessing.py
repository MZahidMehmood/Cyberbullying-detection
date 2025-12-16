import pandas as pd
import re
import unicodedata
import emoji

def normalize_text(text):
    """
    Applies text normalization techniques:
    - NFKC Normalization
    - Emoji to Text conversion (demojize)
    - Lowercasing
    - Collapsing repeated characters (e.g., 'soooo' -> 'soo')
    - Stripping URLs, Mentions, and RTs
    - Preserving Hashtags
    """
    if not isinstance(text, str):
        return ""
    
    # NFKC Normalization
    text = unicodedata.normalize('NFKC', text)
    
    # Emoji to Text
    text = emoji.demojize(text, delimiters=(" ", " "))
    
    # Lowercase
    text = text.lower()
    
    # Strip URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # Strip @mentions
    text = re.sub(r'@\w+', '', text)
    
    # Strip RT (Retweet indicator) - at start or isolated
    text = re.sub(r'^rt\s+', '', text)
    text = re.sub(r'\brt\b', '', text)
    
    # Collapse repeats (keep 2 characters max, e.g., coool -> cool)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_dataframe(df, text_col='tweet_text'):
    """
    Cleans and deduplicates the dataframe.
    """
    print(f"Original shape: {df.shape}")
    
    # Deduplication
    df = df.drop_duplicates(subset=[text_col])
    print(f"Shape after deduplication: {df.shape}")
    
    # Apply normalization
    df['cleaned_text'] = df[text_col].apply(normalize_text)
    
    # Filter empty rows after cleaning
    df = df[df['cleaned_text'].str.len() > 0]
    print(f"Shape after cleaning and removing empty rows: {df.shape}")
    
    return df
