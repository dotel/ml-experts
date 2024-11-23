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

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Dataset preparation
def build_dataset(config, dataset_name="imdb", input_min_text_length=2, input_max_text_length=8):
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




# Advantage calculation (placeholder for GAE or Monte Carlo)
def compute_advantages(rewards):
    return rewards - rewards.mean()


# TRPO surrogate loss
def surrogate_loss(log_probs_old, log_probs_new, advantages):
    ratios = torch.exp(log_probs_new - log_probs_old)
    return -(ratios * advantages).mean()


# Training loop for TRPO
def train_trpo(config):
    # Configurations
    model_name = config.model_name
    batch_size = 1
    num_epochs = 1
    kl_threshold = 0.4  # Trust region threshold

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
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
                logits_old = model(input_ids=input_ids).logits

            # Generate responses
            generated_outputs = model.generate(
                input_ids=input_ids,
                max_length=input_ids.shape[1] + 20,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
            )
            response_ids = generated_outputs[:, input_ids.shape[1]:]

            # Compute new logits
            logits_new = model(input_ids=generated_outputs).logits

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
            rewards = compute_rewards(queries, responses)
            advantages = compute_advantages(rewards)

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
    
    # Save the model and tokenizer
    save_directory = "trpo_model"  
    os.makedirs(save_directory, exist_ok=True)
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    print(f"Model and tokenizer saved to {save_directory}")



if __name__ == "__main__":
    from trl import PPOConfig
    import wandb
    wandb.init()
    config = PPOConfig(
        model_name="distilgpt2",
        learning_rate=1.41e-5,
        log_with="wandb",  # Replace with "wandb" if using logging
    )
    train_trpo(config)
