
""" 
quick_start.py
A simple script to generate text from a trained language model.

It starts generation from a given initial character and produces a sequence of specified
length by repeatedly predicting the next character by choosing the most likely token each step.

Brendan Dileo, July 2025
"""

import torch
from src.models.simple_gpt import SimpleTransformer
from src.data.bpe_tokenizer import BPEDataset
from src.data.load_text import load_training_text
from src.generate.generate import generate_text_bpe

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load and sanitize the training text data
    text = load_training_text(
        "src/data/training.txt", 
        lowercase=True, 
        remove_non_ascii=True, 
        remove_punctuation=True, 
        log_stats=False
    )
    
    # Define block size and embedding dimension
    block_size = 64
    embed_dim = 256
    sp_model_path = "src/data/bpe_tokenizer.model"
    
    # Create dataset
    dataset = BPEDataset(text, block_size, sp_model_path)

    # Recreate model
    model = SimpleTransformer(dataset.vocab_size, embed_dim=embed_dim, block_size=block_size)
    state_dict = torch.load("checkpoints/simple_gpt.pth", map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    
    print("Enter prompts to generate text. Type 'exit' or 'quit' to stop.")
    while True:
        prompt = input("\nPrompt: ")
        if prompt.lower() in ['exit', 'quit']:
            print("Exiting...")
            break
        
        # Generate text
        output_text = generate_text_bpe(
        model=model,
        prompt=prompt,
        max_length=200,
        dataset=dataset,
        device=device,
        method='top_p',
        temperature=0.8,
        top_k=50,
        top_p=0.9
    )

    print(f"\nDecoded Generated text:\n{output_text}")
