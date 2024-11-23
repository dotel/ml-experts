from transformers import AutoTokenizer
import torch
from transformers import AutoTokenizer, GPT2Tokenizer, GPT2LMHeadModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
import torch.nn as nn

class GPT2ActorCriticModel(nn.Module):
    def __init__(self, model_name='gpt2'):
        super(GPT2ActorCriticModel, self).__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained(model_name)
        self.value_head = nn.Linear(self.gpt2.config.n_embd, 1)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.gpt2(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]  # Use the last hidden state
        logits = outputs.logits

        # Compute value estimates
        values = self.value_head(hidden_states).squeeze(-1)
        return logits, values


# Load the fine-tuned model and tokenizer
model_name = "gpt2-imdb-a2c"  # Folder containing the fine-tuned model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = GPT2ActorCriticModel(model_name=model_name).to(device)
model.gpt2.eval()

# Define function to generate responses
def generate_response(prompt, model, tokenizer, max_length=50):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        generated_ids = model.gpt2.generate(
            input_ids=inputs["input_ids"],
            max_length=max_length,
            do_sample=True,  # Enable sampling
            top_k=50,  # Top-k sampling for diversity
            top_p=0.95,  # Nucleus sampling for better quality
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return response

# Prepare prompts
prompts = [
    "The movie was incredibly",
    "I didn't enjoy this film because",
    "One of the most outstanding aspects of the movie was",
    "I hated this movie. This is the worst movie"
]

# Generate and display responses
for prompt in prompts:
    response = generate_response(prompt, model, tokenizer, max_length=50)
    print(f"Prompt: {prompt}\nResponse: {response}\n")
