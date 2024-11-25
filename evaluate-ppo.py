import torch
from datasets import load_dataset
import random
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import pipeline, AutoTokenizer
import torch.nn as nn

# Set a fixed seed for reproducibility
random.seed(42)
base_model = 'lvwerra/gpt2-imdb'
device = 'cuda'  # Use CPU for this example


class GPT2RewardModel(nn.Module):
    def __init__(self, model_name, tokenizer=None):
        super(GPT2RewardModel, self).__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained(model_name)

        # Resize the token embeddings
        if tokenizer is not None:
            self.gpt2.resize_token_embeddings(len(tokenizer))

        self.dropout = nn.Dropout(0.1)
        self.regression_head = nn.Linear(self.gpt2.config.n_embd, 1)
        nn.init.xavier_uniform_(self.regression_head.weight)
        nn.init.zeros_(self.regression_head.bias)

    def forward(self, input_ids, attention_mask=None, apply_sigmoid=False):
        transformer_outputs = self.gpt2.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = transformer_outputs.last_hidden_state
        pooled_output = torch.mean(hidden_states, dim=1)  # Average pooling
        pooled_output = self.dropout(pooled_output)
        logits = self.regression_head(pooled_output).squeeze(-1)
        
        if apply_sigmoid:
            return torch.sigmoid(logits)  # Normalize to [0, 1] for inference
        return logits  # Raw logits for training
reward_tokenizer = GPT2Tokenizer.from_pretrained('./reward-model-imdb-dataset')
reward_tokenizer.pad_token = reward_tokenizer.eos_token
reward_tokenizer.padding_side = 'left'  # Ensure left-padding for GPT-2


# Load the trained reward model
reward_model = GPT2RewardModel(model_name=base_model, tokenizer=reward_tokenizer).to(device)
reward_model.load_state_dict(torch.load('./reward-model-imdb-dataset/reward-model-imdb-dataset.pth', map_location=device))
reward_model.eval()

base_model = 'lvwerra/gpt2-imdb'
def compute_rewards_custom(texts):
    """
    Compute rewards using the trained reward model.
    """
    reward_model.eval()
    rewards = []
    with torch.no_grad():
        for text in texts:
            inputs = reward_tokenizer(
                text,
                truncation=True,
                max_length=256,
                padding='max_length',
                return_tensors='pt',
            ).to(device)

            # Forward pass through the reward model
            outputs = reward_model(inputs['input_ids'], attention_mask=inputs['attention_mask'], apply_sigmoid=True)
            reward = outputs.item()  # Extract the scalar reward
            rewards.append(reward)

    return rewards  # Return a list of scalar rewards

def evaluate_rewards_on_subset(
    model, tokenizer, reward_model, reward_tokenizer, dataset, device='cuda', mod_factor=311
):
    model.eval()
    reward_model.eval()

    subset_texts = [text for i, text in enumerate(dataset) if i % mod_factor == 0]
    generated_texts = []
    rewards = []

    for text in subset_texts:
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=256, padding="max_length"
        ).to(device)

        # Generate response
        with torch.no_grad():
            response = model.generate(
                inputs['input_ids'], max_new_tokens=50, pad_token_id=tokenizer.eos_token_id
            )
            generated_text = tokenizer.decode(response[0], skip_special_tokens=True)
            generated_texts.append(generated_text)

        # Compute reward
        reward = compute_rewards_custom([generated_text])[0]
        rewards.append(reward)

    # Calculate average reward
    avg_reward = sum(rewards) / len(rewards)

    return avg_reward, subset_texts, generated_texts, rewards

# Load test dataset and take a small sample (every 9th example)
test_dataset = load_dataset("imdb", split="test")["text"]

# Load the fine-tuned model and tokenizer
fine_tuned_model = GPT2LMHeadModel.from_pretrained("ppo-trained-model").to(device)
fine_tuned_tokenizer = GPT2Tokenizer.from_pretrained("ppo-trained-model")
fine_tuned_tokenizer.pad_token = reward_tokenizer.eos_token
fine_tuned_tokenizer.padding_side = 'left'  # Ensure left-padding for GPT-2

# Evaluate rewards on the test subset
average_reward, subset_texts, generated_texts, reward_scores = evaluate_rewards_on_subset(
    fine_tuned_model, fine_tuned_tokenizer, reward_model, reward_tokenizer, test_dataset, device
)

# Print the results
print(f"Average Reward on Subset: {average_reward}")
# print("\nSample Results:")
# for i, (input_text, gen_text, reward) in enumerate(zip(subset_texts, generated_texts, reward_scores)):
#     print(f"\nPrompt {i+1}: {input_text}")
#     print(f"Generated Response: {gen_text}")
#     print(f"Reward: {reward}")
