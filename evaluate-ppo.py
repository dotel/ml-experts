import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the fine-tuned model and tokenizer
model_name = "gpt2-imdb-pos-2"  # Replace with the directory where your model is saved
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set the model to evaluation mode
model.eval()

# Move the model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Prepare sample input prompts
prompts = [
    "The movie was incredibly",
    "I didn't enjoy this film because",
    "One of the most outstanding aspects of the movie was",
    "I hated this movie. THis is the worst movie"
]

# Generate responses
response_generation_kwargs = {
    "max_length": 50,  # Adjust this to control the length of the output
    "min_length": 10,
    "do_sample": True,  # Sampling allows more diverse outputs
    "top_k": 50,        # Top-k sampling for diversity
    "top_p": 0.95,      # Nucleus sampling for diversity
    "temperature": 0.7, # Controls randomness
    "pad_token_id": tokenizer.eos_token_id,  # Avoid generation errors
}

for prompt in prompts:
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    output = model.generate(input_ids, **response_generation_kwargs)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    print("=" * 50)
