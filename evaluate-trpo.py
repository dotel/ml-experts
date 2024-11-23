import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Load the saved model and tokenizer
save_directory = "saved_model_trpo"  # Path where the model and tokenizer are saved
model = GPT2LMHeadModel.from_pretrained(save_directory).to(device)
tokenizer = AutoTokenizer.from_pretrained(save_directory)

# Ensure the tokenizer has a padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Function to generate output from the model
def generate_response(prompt, max_length=50):
    # Tokenize the input text
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Generate output
    with torch.no_grad():
        generated_outputs = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode the generated text
    generated_text = tokenizer.decode(generated_outputs[0], skip_special_tokens=True)
    return generated_text


if __name__ == "__main__":
    # Example input prompt
    prompt = "Once upon a time in a faraway land"
    prompts = [
    "The movie was incredibly",
    "I didn't enjoy this film because",
    "One of the most outstanding aspects of the movie was",
    "I hated this movie. This is the worst movie"
    ]
    for prompt in prompts:
        response = generate_response(prompt)
        print(f"Input: {prompt}")
        print(f"Generated Response: {response}")
        print('-' * 50)
