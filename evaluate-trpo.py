import torch
import numpy as np
from transformers import PreTrainedTokenizerFast
from sb3_contrib import TRPO
import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel


# Load the trained RL model
model_path = "gpt2_trpo_model"  # Path to the saved model
rl_model = TRPO.load(model_path)

# Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('./reward_model-2-epoc')

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move the model's policy to evaluation mode and the correct device
policy_model = rl_model.policy
policy_model.eval()
policy_model.to(device)

# Prompts to evaluate
prompts = [
    "The movie was incredibly",
    "I didn't enjoy this film because",
    "One of the most outstanding aspects of the movie was",
    "I hated this movie. This is the worst movie",
]

# Generate responses
max_length = 50  # Maximum number of tokens to generate

for prompt in prompts:
    # Tokenize the initial prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated_tokens = input_ids[0].tolist()  # Initialize with prompt tokens

    for _ in range(max_length - len(input_ids[0])):  # Limit the sequence length
        # Create the observation for the policy
        obs = torch.tensor([generated_tokens[-10:]], dtype=torch.int32).to(device)  # Use the last `max_length` tokens
        with torch.no_grad():
            action = policy_model._predict(obs)  # Predict next token
        
        # Append the predicted token to the generated sequence
        generated_tokens.append(action.item())

        # Stop generation if EOS token is produced
        if action.item() == tokenizer.eos_token_id:
            break

    # Decode the generated sequence to text
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    print("=" * 50)
