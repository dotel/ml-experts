import torch
from tqdm import tqdm
from transformers import AutoTokenizer, GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

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

# Load the trained reward model
class GPT2RewardModel(nn.Module):
    def __init__(self, model_name='gpt2'):
        super(GPT2RewardModel, self).__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.regression_head = nn.Linear(self.gpt2.config.n_embd, 1)

    def forward(self, input_ids, attention_mask=None):
        transformer_outputs = self.gpt2.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = transformer_outputs.last_hidden_state
        pooled_output = hidden_states[:, -1, :]  # Use the representation of the last token
        pooled_output = self.dropout(pooled_output)
        reward = self.regression_head(pooled_output).squeeze(-1)
        return reward

# Instantiate and load the reward model
reward_model = GPT2RewardModel().to(device)
reward_model.load_state_dict(torch.load('./reward_model-2-epoc/reward_model.pth', map_location=device))
reward_model.eval()

# Load the tokenizer for the reward model
reward_tokenizer = GPT2Tokenizer.from_pretrained('./reward_model-2-epoc')

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
            outputs = reward_model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
            reward = outputs.item()  # Extract the scalar reward
            rewards.append(reward)

    return rewards  # Return a list of scalar rewards

def build_dataset(tokenizer, dataset_name="imdb", input_min_text_length=2, input_max_text_length=8):
    # Build a dataset to be used for training.
    tokenizer.pad_token = tokenizer.eos_token
    ds = load_dataset(dataset_name, split="train")
    ds = ds.rename_columns({"text": "review"})
    ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)

    def tokenize(sample):
        input_ids = tokenizer.encode(sample["review"])
        if len(input_ids) > input_max_text_length:
            input_ids = input_ids[:input_max_text_length]
        sample["input_ids"] = input_ids
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch", columns=['input_ids', 'query'])
    return ds

def collator(data):
    input_ids = [torch.tensor(d['input_ids']) for d in data]
    input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    queries = [d['query'] for d in data]
    return {
        'input_ids': input_ids.to(device),
        'attention_mask': attention_mask.to(device),
        'queries': queries,
    }

if __name__ == '__main__':
    # Configuration
    model_name = 'distilgpt2'
    learning_rate = 1e-5
    num_epochs = 1
    batch_size = 40  # Adjust based on your hardware capabilities

    # Load tokenizer and build dataset
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = build_dataset(tokenizer)
    dataset = dataset.select(range(100))  # Limit the dataset size for faster training
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)

    # Instantiate the Actor-Critic model
    model = GPT2ActorCriticModel(model_name=model_name).to(device)
    model.train()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Define gamma for discounting rewards
    gamma = 0.99

    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        epoch_rewards = []
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            queries = batch['queries']

            # Forward pass through the model
            logits, values = model(input_ids=input_ids, attention_mask=attention_mask)

            # Sample actions (next tokens) from the policy
            action_probs = F.softmax(logits[:, -1, :], dim=-1)  # Get probabilities for the last token
            action_distributions = torch.distributions.Categorical(action_probs)
            actions = action_distributions.sample()

            # Generate responses
            responses = []
            for i in range(input_ids.size(0)):  # Use the actual batch size
                generated_ids = model.gpt2.generate(
                    input_ids=input_ids[i].unsqueeze(0),
                    max_length=input_ids.shape[1] + 20,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    pad_token_id=tokenizer.eos_token_id,
                )
                response = tokenizer.decode(generated_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
                responses.append(response)


            # Compute rewards using the reward model
            texts = [q + r for q, r in zip(queries, responses)]
            rewards = compute_rewards_custom(texts)
            rewards = torch.tensor(rewards, dtype=torch.float).to(device)
            epoch_rewards.extend(rewards.cpu().numpy())

            # Compute advantages
            returns = rewards  # For simplicity, consider immediate rewards
            advantages = returns - values[:, -1].detach()

            # Compute actor loss (policy loss)
            log_probs = action_distributions.log_prob(actions)
            actor_loss = - (log_probs * advantages).mean()

            # Compute critic loss (value loss)
            critic_loss = F.mse_loss(values[:, -1].squeeze(-1), returns)

            # Total loss
            loss = actor_loss + critic_loss

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_reward = sum(epoch_rewards) / len(epoch_rewards)
        print(f"Average Reward: {avg_reward:.4f}")
        break

    # Save the fine-tuned model and tokenizer
    model.gpt2.save_pretrained("gpt2-imdb-a2c")
    tokenizer.save_pretrained("gpt2-imdb-a2c")
