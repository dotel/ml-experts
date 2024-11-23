from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load pretrained GPT-2 model and tokenizer
model_name = "gpt2"  # Use "gpt2-medium" or "gpt2-large" for larger variants
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
from transformers import PreTrainedTokenizerFast
from sb3_contrib import TRPO
import os



tokenizer_dir="tokenizer_5k"
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=os.path.join(tokenizer_dir, 'tokenizer.json'),
    unk_token="<unk>",
    pad_token="<pad>",
    cls_token="<s>",
    sep_token="</s>",
    mask_token="<mask>",
)
tokenizer.pad_token = tokenizer.unk_token  # Set padding token

model = GPT2LMHeadModel.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))  # Resize embeddings after loading weights

# Move model to evaluation mode and set device
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

# Prompts to evaluate
prompts = [
    "The movie was incredibly",
    "I didn't enjoy this film because",
    "One of the most outstanding aspects of the movie was",
    "I hated this movie. This is the worst movie",
]

# Parameters for text generation
generation_kwargs = {
    "max_length": 50,  # Maximum length of generated text (including prompt)
    "do_sample": True,  # Enable sampling (vs. greedy decoding)
    "top_k": 50,  # Limit to top-k tokens for sampling
    "top_p": 0.95,  # Use nucleus sampling with cumulative probability
    "temperature": 0.7,  # Adjust randomness of sampling
    "pad_token_id": tokenizer.eos_token_id,  # Ensure padding works properly
}

# Generate and print responses for each prompt
for prompt in prompts:
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Generate text
    with torch.no_grad():
        outputs = model.generate(input_ids, **generation_kwargs)

    # Decode and print the generated text
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    print("=" * 50)
