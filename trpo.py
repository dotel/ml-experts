import torch
from tqdm import tqdm
from transformers import AutoTokenizer, GPT2LMHeadModel
from datasets import load_dataset
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from trl.core import LengthSampler
import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel


# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
base_model = 'lvwerra/gpt2-imdb'


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


# Load the trained reward model
reward_model = GPT2RewardModel(model_name=base_model, tokenizer=reward_tokenizer).to(device)
reward_model.load_state_dict(torch.load('./reward-model-imdb-dataset/reward-model-imdb-dataset.pth', map_location=device))
reward_model.eval()

# Load the tokenizer

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


# Dataset preparation
def build_dataset(config, dataset_name="imdb", input_min_text_length=16, input_max_text_length=64):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    ds = load_dataset(dataset_name, split="train")
    ds = ds.rename_columns({"text": "review"})
    ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)

    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["review"])[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds


def collator(data, tokenizer):
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    input_ids = [torch.tensor(d['input_ids'], dtype=torch.long) for d in data]
    input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    queries = [d['query'] for d in data]
    return {
        'input_ids': input_ids.to(device),
        'attention_mask': attention_mask.to(device),
        'queries': queries,
    }


# Reward function (placeholder example)
def compute_rewards(queries, responses):
    # Positive reward for responses with sentiment keywords
    rewards = []
    for query, response in zip(queries, responses):
        reward = 1.0 if "good" in response.lower() else -1.0
        rewards.append(reward)
    return torch.tensor(rewards, device=device)


# KL-divergence calculation
def compute_kl_divergence(logits_old, logits_new, attention_mask):
    # Slice logits_new to match the sequence length of logits_old
    logits_new = logits_new[:, :logits_old.size(1), :]  # Ensure matching sequence lengths

    # Compute probabilities
    probs_old = F.softmax(logits_old, dim=-1)
    probs_new = F.softmax(logits_new, dim=-1)

    # Calculate KL-divergence
    kl = F.kl_div(probs_new.log(), probs_old, reduction='none')

    # Apply attention mask to calculate mean KL-divergence
    return (kl.sum(dim=-1) * attention_mask).mean()

# Define the Actor-Critic Model
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



# Advantage calculation (placeholder for GAE or Monte Carlo)
def compute_advantages(rewards):
    return rewards - rewards.mean()


# TRPO surrogate loss
def surrogate_loss(log_probs_old, log_probs_new, advantages):
    ratios = torch.exp(log_probs_new - log_probs_old)
    print(f"Ratios: {ratios.shape}, Advantages: {advantages}")
    return -(ratios * advantages).mean()


# Training loop for TRPO
def train_trpo(config):
    # Configurations
    model_name = config.model_name
    batch_size = 1
    num_epochs = 200
    kl_threshold = 0.5  # Trust region threshold

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = GPT2ActorCriticModel(model_name=model_name).to(device)
    model.train()

    # Build dataset and dataloader
    dataset = build_dataset(config)
    # Reduce the dataset
    dataset = dataset.select(range(100))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: collator(x, tokenizer))

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            queries = batch['queries']

            with torch.no_grad():
                logits_old, values_old = model(input_ids=input_ids)

            # Generate responses
            generated_outputs = model.gpt2.generate(
                input_ids=input_ids,
                max_length=input_ids.shape[1] + 20,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
            )
            response_ids = generated_outputs[:, input_ids.shape[1]:]

            # Compute new logits
            logits_new, values = model(input_ids=generated_outputs)


           # Compute log probabilities
            response_log_probs_old = F.log_softmax(logits_old, dim=-1)
            response_log_probs_new = F.log_softmax(logits_new, dim=-1)

            # Align shapes of response_ids and logits
            response_ids_expanded = response_ids.unsqueeze(-1)

            # Truncate or pad to match dimensions
            min_len = min(response_log_probs_old.size(1), response_ids_expanded.size(1))
            response_log_probs_old = response_log_probs_old[:, :min_len, :]
            response_log_probs_new = response_log_probs_new[:, :min_len, :]
            response_ids_expanded = response_ids_expanded[:, :min_len, :]

            # Gather log probabilities
            response_log_probs_old = response_log_probs_old.gather(2, response_ids_expanded).squeeze(-1)
            response_log_probs_new = response_log_probs_new.gather(2, response_ids_expanded).squeeze(-1)

            # Compute rewards and advantages
            responses = [tokenizer.decode(ids) for ids in response_ids]
            texts = [q + r for q, r in zip(queries, responses)]
            rewards = compute_rewards_custom(texts)  # Returns a list of scalars

            # returns = rewards  # For simplicity, consider immediate rewards
            returns = torch.tensor(rewards, dtype=torch.float32, device=values.device)

            advantages = returns - values[:, -1].detach()


            # Compute surrogate loss
            loss = surrogate_loss(response_log_probs_old, response_log_probs_new, advantages)

            print(f"logits_old: {logits_old.shape}, logits_new: {logits_new.shape}")
            print(f"response_ids_expanded: {response_ids_expanded.shape}")
            print(f"response_log_probs_old: {response_log_probs_old.shape}")
            print(f"response_log_probs_new: {response_log_probs_new.shape}")
            print(f"Aligned logits_new: {logits_new.shape}, logits_old: {logits_old.shape}")
            print(f"Attention mask: {attention_mask.shape}")



            # KL-divergence constraint
            kl = compute_kl_divergence(logits_old, logits_new, attention_mask)
            if kl > kl_threshold:
                print(f"KL-divergence too high: {kl.item()}, skipping update")
                continue

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Loss: {loss.item()}, KL-divergence: {kl.item()}")
            wandb.log({
                "epoch": epoch + 1,
                "loss": loss.item(),
                "kl_divergence": kl.item(),
                # "mean_reward": rewards.mean().item(),
            })
    
    # Save the model and tokenizer
    save_directory = "trpo_model"  
    os.makedirs(save_directory, exist_ok=True)
    model.gpt2.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    print(f"Model and tokenizer saved to {save_directory}")



if __name__ == "__main__":
    from trl import PPOConfig
    import wandb
    config = PPOConfig(
        model_name="distilgpt2",
        learning_rate=1.41e-5,
        log_with="wandb",  # Replace with "wandb" if using logging
    )
    wandb.init(project="TRPO_Training", config=config.__dict__)
    
    
    train_trpo(config)

    wandb.finish()
