import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, GPT2LMHeadModel
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel

import os

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
base_model = 'lvwerra/gpt2-imdb'

# Load the tokenizer and model
model_name = 'gpt2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

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
        hidden_states = outputs.hidden_states[-1]  # Last hidden state
        logits = outputs.logits
        values = self.value_head(hidden_states).squeeze(-1)
        return logits, values

model = GPT2ActorCriticModel(model_name=model_name).to(device)

# Build the dataset
def build_dataset(input_min_text_length=16, input_max_text_length=64):
    ds = load_dataset("imdb", split="train")
    ds = ds.rename_columns({"text": "review"})
    ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)
    input_size = np.random.randint(input_min_text_length, input_max_text_length+1)

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["review"])[:input_size]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds

def collator(data):
    input_ids = [torch.tensor(d['input_ids'], dtype=torch.long) for d in data]
    input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    queries = [d['query'] for d in data]
    return {
        'input_ids': input_ids.to(device),
        'attention_mask': attention_mask.to(device),
        'queries': queries,
    }




# Reward function using the trained reward model
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
    return torch.tensor(rewards, dtype=torch.float32, device=device)


# Helper functions for TRPO
def get_flat_params_from(model):
    return torch.cat([param.data.view(-1) for param in model.parameters()])

def set_flat_params_to(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = param.numel()
        param.data.copy_(flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size

def get_flat_grad_from(loss, model):
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True, allow_unused=True)
    return torch.cat([grad.view(-1) for grad in grads])

def conjugate_gradients(Avp, b, nsteps, residual_tol=1e-10):
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)

    for _ in range(nsteps):
        Avp_p = Avp(p)
        alpha = rdotr / torch.dot(p, Avp_p)
        x += alpha * p
        r -= alpha * Avp_p
        new_rdotr = torch.dot(r, r)
        if new_rdotr < residual_tol:
            break
        beta = new_rdotr / rdotr
        p = r + beta * p
        rdotr = new_rdotr
    return x

def line_search(model, f, x, fullstep, expected_improve_rate, max_backtracks=10, accept_ratio=0.1):
    fval = f().data
    for (stepfrac) in [.5**x for x in range(max_backtracks)]:
        xnew = x + stepfrac * fullstep
        set_flat_params_to(model, xnew)
        newfval = f().data
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve
        if ratio.item() > accept_ratio and actual_improve.item() > 0:
            print(f"Accepted step size: {stepfrac}")
            return True
    return False

def flatten_parameters(model):
    return torch.cat([param.view(-1) for param in model.parameters()])

def compute_kl_divergence(new_log_probs, old_log_probs):
    kl = old_log_probs.exp() * (old_log_probs - new_log_probs)
    return kl.mean()

def fisher_vector_product(model, states, actions, old_log_probs, v, damping=1e-2):
    def get_kl():
        logits, _ = model(states)
        new_log_probs = F.log_softmax(logits, dim=-1)
        new_action_log_probs = new_log_probs.gather(2, actions.unsqueeze(-1)).squeeze(-1)
        kl = (old_log_probs - new_action_log_probs).mean()
        return kl

    kl = get_kl()
    grads = torch.autograd.grad(kl, model.parameters(), create_graph=True)
    flat_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads])

    kl_v = (flat_grad_kl * v).sum()
    grads = torch.autograd.grad(kl_v, model.parameters())
    flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads])

    return flat_grad_grad_kl + damping * v

# Training loop for TRPO
def train_trpo():
    batch_size = 8
    num_epochs = 1
    max_kl = 1e-2  # Trust region constraint

    # Build dataset and dataloader
    dataset = build_dataset()
    dataset = dataset.select(range(100))  # Subset for demonstration
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            queries = batch['queries']

            # Generate responses
            with torch.no_grad():
                generated_outputs = model.gpt2.generate(
                    input_ids=input_ids,
                    max_length=input_ids.shape[1] + 20,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    pad_token_id=tokenizer.eos_token_id,
                )

            full_input_ids = generated_outputs  # Concatenated input and response

            # Compute old action log probabilities
            with torch.no_grad():
                logits_old, values_old = model(full_input_ids)
                log_probs_old = F.log_softmax(logits_old, dim=-1)
                actions = full_input_ids
                action_log_probs_old = log_probs_old.gather(2, actions.unsqueeze(-1)).squeeze(-1)
                action_log_probs_old = action_log_probs_old.sum(dim=1)

            # Compute rewards and advantages
            responses = [tokenizer.decode(ids[input_ids.shape[1]:]) for ids in generated_outputs]
            texts = [q + r for q, r in zip(queries, responses)]
            rewards = compute_rewards_custom(texts)
            _, values = model(full_input_ids)
            values = values[:, -1]  # Use the last value
            advantages = rewards - values.detach()

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Define surrogate loss
            def surrogate_loss():
                logits_new, _ = model(full_input_ids)
                log_probs_new = F.log_softmax(logits_new, dim=-1)
                action_log_probs_new = log_probs_new.gather(2, actions.unsqueeze(-1)).squeeze(-1)
                action_log_probs_new = action_log_probs_new.sum(dim=1)
                ratio = torch.exp(action_log_probs_new - action_log_probs_old)
                surrogate = (ratio * advantages).mean()
                return -surrogate

            # Compute gradients of surrogate loss
            loss = surrogate_loss()
            grads = get_flat_grad_from(loss, model).detach()

            # Compute step direction
            def Fvp(v):
                return fisher_vector_product(model, full_input_ids, actions, action_log_probs_old.detach(), v)

            stepdir = conjugate_gradients(Fvp, grads, 10)

            # Compute step size
            shs = 0.5 * (stepdir * Fvp(stepdir)).sum(0, keepdim=True)
            lm = torch.sqrt(shs / max_kl)
            fullstep = stepdir / lm[0]

            # Compute expected improvement rate
            expected_improve = (grads * fullstep).sum(0, keepdim=True)

            # Save current parameters
            prev_params = get_flat_params_from(model)

            # Line search to find the best step size
            success = line_search(
                model,
                surrogate_loss,
                prev_params,
                fullstep,
                expected_improve,
            )

            if not success:
                set_flat_params_to(model, prev_params)
                print("Line search failed. Using previous parameters.")
            else:
                print("Line search succeeded.")

            # Update value function (critic)
            # Using MSE loss between predicted values and rewards
            value_loss_fn = nn.MSELoss()
            value_optimizer = torch.optim.Adam(model.value_head.parameters(), lr=1e-3)

            for _ in range(5):  # Number of value function updates
                value_optimizer.zero_grad()
                _, values = model(full_input_ids)
                values = values[:, -1]
                value_loss = value_loss_fn(values, rewards)
                value_loss.backward()
                value_optimizer.step()

    # Save the model and tokenizer
    save_directory = "trpo_model"
    os.makedirs(save_directory, exist_ok=True)
    model.gpt2.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    print(f"Model and tokenizer saved to {save_directory}")

if __name__ == "__main__":
    train_trpo()
