import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the fine-tuned policy model and tokenizer
model_name = "./policy_model-2-epoch"  # Replace with your trained model directory
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define a pad token if it hasn't been set
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '<|pad|>'})

# Load the model and resize token embeddings to accommodate new tokens
model = AutoModelForCausalLM.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))

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
    "I hated this movie. This is the worst movie"
]

# Configuration for text generation
response_generation_kwargs = {
    "max_length": 50,  # Maximum output length
    "min_length": 10,  # Minimum output length
    "do_sample": True,  # Enables sampling for diversity
    "top_k": 40,        # Limits token sampling to top-k probabilities
    "top_p": 0.85,      # Nucleus sampling for diversity
    "temperature": 0.7, # Reduces randomness
    "repetition_penalty": 1.2,  # Penalizes repetition
    "pad_token_id": tokenizer.pad_token_id,  # Explicitly set the pad token ID
}

# Generate and display responses
print("Generated Responses:")
for prompt in prompts:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=50  # Ensure inputs fit within the model's max sequence length
    ).to(device)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    # Generate responses
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,  # Pass attention mask to handle padding correctly
        **response_generation_kwargs
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    print("=" * 50)
