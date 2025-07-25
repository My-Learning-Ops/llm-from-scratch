
""" 
load_text.py
This module provides a function to load and sanitize text data for training.


Brendan Dileo, July 2025
"""


def load_training_text(file_path: str) -> str:
    
    with open(file_path, 'r', encoding='utf-8') as file:
        raw = file.read()
    
    sanitized = raw.strip().replace('\r\n', '\n')
    sanitized = ' '.join(sanitized.split())
    return sanitized
        
    