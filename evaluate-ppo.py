import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the fine-tuned model and tokenizer
model_name = "ppo-trained-model"  # Replace with the directory where your model is saved
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
    "max_length": 50,
    "min_length": 30,
    "do_sample": True,
    "top_k": 50,
    "top_p": 0.9,  # Enable nucleus sampling
    "temperature": 0.7,  # Increase temperature for more diversity
    "pad_token_id": tokenizer.eos_token_id,
}

for prompt in prompts:
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    output = model.generate(input_ids, **response_generation_kwargs)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    print("=" * 50)
