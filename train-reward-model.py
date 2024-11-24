from datasets import load_dataset
import os
import torch
import torch.nn as nn
import wandb

import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel
)

from transformers import GPT2Tokenizer

from transformers import PreTrainedTokenizerFast

from datasets import load_dataset, DatasetDict

from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

import numpy as np
from tqdm import tqdm

device = 'cuda'
print(f'Using device: {device}')
ds = load_dataset("ajaykarthick/imdb-movie-reviews")
base_model = 'lvwerra/gpt2-imdb'

from torch.utils.data import Dataset
tokenizer = GPT2Tokenizer.from_pretrained(base_model)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

class RLHFDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=256):
        # print(dataset, ' is the dataset', dataset[0])
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.texts = []
        self.scores = []

        for example in dataset:
            chosen_text = example['review']
            self.texts.append(chosen_text)
            if example['label'] == 0:
                self.scores.append(1.0)
            else:
                self.scores.append(0.0)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        score = self.scores[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt',
        )

        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['score'] = torch.tensor(score, dtype=torch.float)
        return item



# Combine train and test splits
from datasets import concatenate_datasets

full_dataset = concatenate_datasets([ds['train'], ds['test']])

# Use only a subset of the dataset
# subset_size = 1000  # Adjust this number based on your resource constraints
# full_dataset = full_dataset.select(range(subset_size))

# Create the RLHF dataset
rlhf_dataset = RLHFDataset(full_dataset, tokenizer)

# Split into training and validation sets
from torch.utils.data import random_split

total_length = len(rlhf_dataset)
train_length = int(0.9 * total_length)
val_length = total_length - train_length

train_dataset, val_dataset = random_split(rlhf_dataset, [train_length, val_length])

# Create DataLoaders
from torch.utils.data import DataLoader
batch_size = 8

train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size)



import torch.nn as nn
from transformers import GPT2LMHeadModel

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


# Initialize the model with the new tokenizer
reward_model = GPT2RewardModel(tokenizer=tokenizer, model_name=base_model).to(device)

optimizer = optim.AdamW(reward_model.parameters(), lr=5e-6) #Prev 1e-5 
loss_fn = nn.BCEWithLogitsLoss()


from tqdm.auto import tqdm

def train_reward_model(model, train_loader, val_loader, optimizer, loss_fn, epochs):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        running_loss = 0  # For smoothing
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
        for batch in progress_bar:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            scores = batch['score'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, scores)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            running_loss += loss.item()  # Smooth loss tracking
            progress_bar.set_postfix(loss=running_loss / (progress_bar.n + 1))

            # Log training loss to wandb
            wandb.log({"train_loss": loss.item()})

            # Evaluate validation loss more frequently
            if (progress_bar.n + 1) % 500 == 0:  # Every 500 steps
                val_loss = 0
                model.eval()
                with torch.no_grad():
                    for val_batch in val_loader:
                        val_input_ids = val_batch['input_ids'].to(device)
                        val_attention_mask = val_batch['attention_mask'].to(device)
                        val_scores = val_batch['score'].to(device)

                        val_outputs = model(val_input_ids, val_attention_mask)
                        val_loss += loss_fn(val_outputs, val_scores).item()
                val_loss /= len(val_loader)
                print(f"Step {progress_bar.n + 1}: Validation Loss = {val_loss}")
                wandb.log({"val_loss_step": val_loss})
                model.train()  # Return to training mode

        # End of epoch validation
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                scores = batch['score'].to(device)

                outputs = model(input_ids, attention_mask)
                loss = loss_fn(outputs, scores)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f'Validation Loss: {val_loss}')

        # Log validation loss to wandb
        wandb.log({"val_loss": val_loss, "epoch": epoch + 1})

wandb.init(
    project="rlhf-reward-model",  # Replace with your project name
    config={
        "learning_rate": 1e-2,
        "batch_size": batch_size,
        "epochs": 5,
        "model_name": "gpt2",
    }
)

# Train the reward model
train_reward_model(reward_model, train_loader, val_loader, optimizer, loss_fn, epochs=3)

# Save the trained reward model   
torch.save(reward_model.state_dict(), './reward-model-imdb-dataset.pth')

#reward_model.save_pretrained('./reward_model')
tokenizer.save_pretrained('./reward-model-imdb-dataset')

wandb.finish()

