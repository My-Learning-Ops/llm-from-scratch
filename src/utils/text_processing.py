
""" 
text_processing.py


This module provides a function to load and sanitize text data for training.

Brendan Dileo, July 2025
"""

import string


def load_training_text(
    file_path: str,
    lowercase: bool = True,
    remove_non_ascii: bool = True,
    remove_punctuation: bool = True,
    log_stats: bool = True,
    max_length: int = None,
) -> str:
    """ Loads training text data from a file and sanitizes it.

    Args:
        file_path (str): _description_
        lowercase (bool, optional): _description_. Defaults to True.
        remove_non_ascii (bool, optional): _description_. Defaults to True.
        remove_punctuation (bool, optional): _description_. Defaults to True.
        log_stats (bool, optional): _description_. Defaults to True.
        max_length (int, optional): _description_. Defaults to None.

    Returns:
        str: _description_
    """
    
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    original_length = len(text)
    
    # Normalize line endings and collapse any whitespace
    text = text.replace('\r\n', '\n')
    text = ' '.join(text.strip().split())
    
    # Convert to lowercase if specified
    if lowercase:
        text = text.lower()
    
    # Remove non ASCII characters if specified
    if remove_non_ascii:
        text = text.encode('ascii', 'ignore').decode()
    
    # Remove punctuation if specified
    if remove_punctuation:
        text = text.translate(str.maketrans('', '', string.punctuation))
        
    # Limit the length of the text if specified
    if max_length is not None:
        text = text[:max_length]
    
    # Log statistics if enabled
    if log_stats:
        print(f"[INFO] Loaded: {file_path}")
        print(f"[INFO] Original length: {original_length:,} characters")
        print(f"[INFO] Sanitized length: {len(text):,} characters")
        unique_chars = len(set(text))
        print(f"[INFO] Unique characters: {unique_chars}")
    
    return text
