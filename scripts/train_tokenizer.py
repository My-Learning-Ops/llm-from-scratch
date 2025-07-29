
""" 
train_tokenizer.py
Defines a script to train a Byte Pair Encoding (BPE) tokenizer using SentencePiece.
Aligned with BPEDataset for consistent tokenization in training & inference.

Brendan Dileo, July 2025
"""

import os
import logging
import sentencepiece as spm
from src.config.config import MODEL_CONFIG

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    input_file = "src/data/training.txt"
    model_prefix = "src/data/bpe_tokenizer"
    vocab_size = MODEL_CONFIG['vocab_size']

    model_file = f"{model_prefix}.model"
    if os.path.exists(model_file):
        print(f"Tokenizer model {model_file} already exists. Skipping training.")
    else:
        spm.SentencePieceTrainer.Train(
            input=input_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type='bpe',
            character_coverage=1.0,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3
        )
        print(f"Trained BPE tokenizer with vocab size {vocab_size} and saved to {model_file}")