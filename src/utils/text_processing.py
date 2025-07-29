
""" 
text_processing.py

This module provides a function to load and sanitize text data for training.

Brendan Dileo, July 2025
"""

import string
import re


def load_training_text(
    file_path: str,
    lowercase: bool = False,
    remove_non_ascii: bool = True,
    remove_punctuation: bool = False,
    preserve_sentences: bool = True,
    log_stats: bool = True,
    max_length: int = None,
) -> str:
    """ Loads training text data from a file and sanitizes it.

    file_path (str): Path to the text file to load
        lowercase (bool, optional): Convert text to lowercase. Defaults to False.
        remove_non_ascii (bool, optional): Remove non-ASCII characters. Defaults to True.
        remove_punctuation (bool, optional): Remove all punctuation. Defaults to False.
        preserve_sentences (bool, optional): Keep sentence structure with proper spacing. Defaults to True.
        log_stats (bool, optional): Print loading statistics. Defaults to True.
        max_length (int, optional): Maximum text length to keep. Defaults to None.
        
    Returns:
        str: Cleaned and processed text
    """
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
    except UnicodeDecodeError:
        # Fallback encoding if UTF-8 fails
        with open(file_path, 'r', encoding='latin-1') as file:
            text = file.read()
            if log_stats:
                print(f"[WARNING] Used latin-1 encoding for {file_path}")
    
    
    original_length = len(text)
    
    # Better whitespace normalization
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    if preserve_sentences:
        # Keep paragraph breaks but normalize other whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        # Normalize spaces and tabs
        text = re.sub(r'[ \t]+', ' ', text)
        # Clean around newlines
        text = re.sub(r' *\n *', '\n', text)
    else:
        # Original behavior - collapse all whitespace
        text = ' '.join(text.strip().split())
    
    # Convert to lowercase if specified
    if lowercase:
        text = text.lower()
    
    # Remove non-ASCII characters if specified
    if remove_non_ascii:
        text = text.encode('ascii', 'ignore').decode()
    
    # Improved punctuation handling
    if remove_punctuation:
        if preserve_sentences:
            # Keep sentence ending punctuation, remove others
            keep_punct = '.!?'
            remove_punct = ''.join(c for c in string.punctuation if c not in keep_punct)
            text = text.translate(str.maketrans('', '', remove_punct))
            # Fix spacing around kept punctuation
            text = re.sub(r'([.!?])\s*', r'\1 ', text)
        else:
            # Remove all punctuation (original behavior)
            text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Clean up extra whitespace that might have been introduced
    if preserve_sentences:
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n\n+', '\n\n', text)
    else:
        text = ' '.join(text.split())
    
    # Remove very short lines
    if preserve_sentences:
        lines = text.split('\n')
        filtered_lines = []
        for line in lines:
            line = line.strip()
            if len(line) >= 3 or line == '':
                filtered_lines.append(line)
        text = '\n'.join(filtered_lines)
    
    # Limit the length of the text if specified
    if max_length is not None:
        text = text[:max_length]
    
    text = text.strip()
    
    # Stats logging
    if log_stats:
        print(f"[INFO] Loaded: {file_path}")
        print(f"[INFO] Original length: {original_length:,} characters")
        print(f"[INFO] Processed length: {len(text):,} characters")
        print(f"[INFO] Reduction: {(1 - len(text)/original_length)*100:.1f}%")
        
        unique_chars = len(set(text))
        print(f"[INFO] Unique characters: {unique_chars}")
        
        # Show character distribution
        if unique_chars <= 100:  # Only for reasonable sizes
            char_sample = ''.join(sorted(set(text))[:50])
            print(f"[INFO] Character sample: {repr(char_sample)}")
        
        # Show text sample
        sample_length = min(100, len(text))
        if sample_length > 0:
            sample = text[:sample_length].replace('\n', '\\n')
            print(f"[INFO] Text sample: {repr(sample)}...")
    
    return text