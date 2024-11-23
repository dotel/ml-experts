import torch
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

class GPT2RewardModel(nn.Module):
    def __init__(self, model_name='gpt2'):
        super(GPT2RewardModel, self).__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.regression_head = nn.Linear(self.gpt2.config.n_embd, 1)
    
    def forward(self, input_ids, attention_mask=None):
        outputs = self.gpt2.transformer(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        pooled_output = hidden_states[:, -1, :]  # Use the last token's hidden state
        pooled_output = self.dropout(pooled_output)
        reward = self.regression_head(pooled_output).squeeze(-1)
        return reward

# Load the trained reward model
reward_model = GPT2RewardModel().to(device)
reward_model.load_state_dict(torch.load('./reward_model-2-epoc/reward_model.pth', map_location=device))
reward_model.eval()

# Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

def compute_rewards_custom(texts):
    """
    Compute rewards using the trained reward model.
    """
    reward_model.eval()
    rewards = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(
                text,
                truncation=True,
                max_length=256,
                padding='max_length',
                return_tensors='pt',
            ).to(device)
            outputs = reward_model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
            reward = outputs.item()  # Extract the scalar reward
            rewards.append(reward)
    return rewards  # Return a list of scalar rewards

def build_dataset(input_min_text_length=2, input_max_text_length=8):
    # Build a dataset to be used for the training.
    ds = load_dataset("imdb", split="train")
    ds = ds.rename_columns({"text": "review"})
    ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)

    input_size_sampler = lambda: torch.randint(input_min_text_length, input_max_text_length + 1, (1,)).item()

    def tokenize(sample):
        input_size = input_size_sampler()
        sample["input_ids"] = tokenizer.encode(sample["review"])[:input_size]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

if __name__ == '__main__':
    # Load the dataset
    dataset = build_dataset()

    # Initialize the policy model (GPT-2)
    policy_model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    policy_model.resize_token_embeddings(len(tokenizer))

    # Set up the optimizer
    optimizer = optim.Adam(policy_model.parameters(), lr=1e-5)

    # Configuration for text generation
    output_min_length = 4
    output_max_length = 16
    output_length_sampler = lambda: torch.randint(output_min_length, output_max_length + 1, (1,)).item()

    # Training loop
    num_epochs = 2
    batch_size = 4  # Adjust based on your GPU memory
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        for batch in tqdm(dataloader):
            optimizer.zero_grad()

            # Prepare inputs
            query_texts = batch["query"]
            query_tensors = [torch.tensor(t).to(device) for t in batch["input_ids"]]

            # Generate responses
# Generate responses
            response_texts = []
            log_probs = []
            for query_tensor in query_tensors:
                gen_len = output_length_sampler()
                input_ids = query_tensor.unsqueeze(0)
                attention_mask = torch.ones_like(input_ids).to(device)
                outputs = policy_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=gen_len,
                    do_sample=True,
                    top_k=0,
                    top_p=1.0,
                    pad_token_id=tokenizer.eos_token_id,
                )
                response = outputs[0][input_ids.shape[1]:]
                response_text = tokenizer.decode(response, skip_special_tokens=True)
                response_texts.append(response_text)

                # Calculate log-probabilities without torch.no_grad()
                full_input_ids = torch.cat([input_ids[0], response], dim=0)
                outputs = policy_model(full_input_ids.unsqueeze(0))
                logits = outputs.logits
                log_prob = torch.nn.functional.log_softmax(logits, dim=-1)
                selected_log_probs = log_prob[0, range(len(input_ids[0]) - 1, len(full_input_ids) - 1), full_input_ids[len(input_ids[0]):]]
                log_probs.append(selected_log_probs.sum())


            # Compute rewards
            texts = [q + r for q, r in zip(query_texts, response_texts)]
            rewards = compute_rewards_custom(texts)  # Returns a list of scalars

            # Normalize rewards
            rewards_tensor = torch.tensor(rewards, dtype=torch.float).to(device)
            rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)

            # Compute policy loss
            policy_loss = []
            for log_prob, reward in zip(log_probs, rewards_tensor):
                loss = -log_prob * reward  # Negative for gradient ascent
                policy_loss.append(loss)
            policy_loss = torch.stack(policy_loss).mean()

            # Backpropagation
            policy_loss.backward()
            optimizer.step()

    # Save the model
    torch.save(policy_model.state_dict(), f'policy_model-{epoch+1}-epoch.pth')
    tokenizer.save_pretrained(f'policy_model-{epoch+1}-epoch')
    save_dir = "./trained_policy_model"
    policy_model.save_pretrained(save_dir)
