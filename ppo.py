import torch
from tqdm import tqdm

from transformers import pipeline, AutoTokenizer
from datasets import load_dataset
import wandb

from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch.nn as nn
import warnings

# device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
device = 'cuda'  # Use CPU for this example
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

def build_dataset(config, dataset_name="imdb", input_min_text_length=2, input_max_text_length=8):
    # Build a dataset to be used for the training.
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    ds = load_dataset(dataset_name, split="train")
    # use smaller ds for test
    ds = ds.rename_column("text", "review")
    ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)

    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["review"])[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

if __name__ == '__main__':
    config = PPOConfig(
        model_name="lvwerra/gpt2-imdb",
        learning_rate=1.41e-5,
        log_with="wandb",
    )

    wandb.init()

    dataset = build_dataset(config)

    # This is the model we are going to fine-tune with PPO
    model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
    # This is the reference model (frozen) for the KL divergence
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator)

    # The configuration to generate responses (trajectories)
    output_min_length = 4
    output_max_length = 16
    output_length_sampler = LengthSampler(output_min_length, output_max_length)

    response_generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
    }

    import time
    total_batches = len(ppo_trainer.dataloader)
    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        start = time.time()

        print("*"*50)
        query_tensors = batch["input_ids"]
        
        #### Phase 1: Get trajectories from the offline policy
        response_tensors = []
        for query in query_tensors:
            gen_len = output_length_sampler()
            response_generation_kwargs["max_new_tokens"] = gen_len
            response = ppo_trainer.generate(query, **response_generation_kwargs)
            response_tensors.append(response.squeeze()[-gen_len:])
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

        #### Phase 1: Compute rewards
        
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        rewards = compute_rewards_custom(texts)  # Returns a list of scalars

        # Convert rewards to a tensor for normalization
        rewards_tensor = torch.tensor(rewards, dtype=torch.float).to(device)

        # Normalize rewards between 0 and 1
        rewards_tensor = (rewards_tensor - rewards_tensor.min()) / (rewards_tensor.max() - rewards_tensor.min() + 1e-8)

        # Convert normalized rewards back to a list of tensors
        rewards = [torch.tensor(r, dtype=torch.float).to(device) for r in rewards_tensor.tolist()]

        #### Phase 1 + Phase 2: calculate the logprobs and then run the PPO update
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)
        end = time.time()
        print(end - start)
        # print(f"Total skipped batches: {skipped_batches}")
        # print(f"Total batches: {total_batches}")

    model.save_pretrained("ppo-trained-model", push_to_hub=False)
    tokenizer.save_pretrained("ppo-trained-model", push_to_hub=False)

    print("Training finished.")
    # print(f"Total skipped batches: {skipped_batches}")
    print(f"Total batches: {total_batches}")

