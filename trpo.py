import gym
import torch
import torch.nn as nn
from gym import spaces
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset
from sb3_contrib import TRPO
import numpy as np
from transformers import PreTrainedTokenizerFast
from transformers import AutoConfig, GPT2LMHeadModel
import os

from transformers import GPT2LMHeadModel, AutoTokenizer
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from torch.cuda.amp import autocast


# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# tokenizer_dir="tokenizer_5k"
tokenizer = GPT2Tokenizer.from_pretrained('./reward_model-2-epoc')
# tokenizer.pad_token = tokenizer.unk_token  # Set padding token

class GPT2RewardModel(nn.Module):
    def __init__(self):
        super(GPT2RewardModel, self).__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')
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

# Load the trained reward model
reward_model = GPT2RewardModel().to('cpu')
# reward_model.gpt2.resize_token_embeddings(len(tokenizer))  # Resize embeddings after loading weights

reward_model.load_state_dict(torch.load('./reward_model-2-epoc/reward_model.pth', map_location=device))
reward_model.eval()


class LanguageModelEnv(gym.Env):
    def __init__(self, tokenizer, reward_model, max_length=10):
        super(LanguageModelEnv, self).__init__()
        self.tokenizer = tokenizer
        self.reward_model = reward_model
        self.max_length = max_length
        self.vocab_size = len(tokenizer)

        # Define action and observation spaces
        self.action_space = spaces.Discrete(self.vocab_size)
        self.observation_space = spaces.Box(
            low=0,
            high=self.vocab_size - 1,
            shape=(self.max_length,),
            dtype=np.int32,
        )

    def reset(self):
        # Reset the generated sequence
        self.generated = []

        # Initialize current observation as a sequence of padding tokens
        self.current_obs = [self.tokenizer.pad_token_id] * self.max_length

        # Return the observation as a numpy array
        return np.array(self.current_obs, dtype=np.int32)
    
    def compute_reward(self, text):
        # Use the reward model to compute the reward
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=256,
            padding='max_length',
            return_tensors='pt',
        )
        inputs = {key: val.to("cpu") for key, val in inputs.items()}  # Move inputs to CPU
        with torch.no_grad():
            outputs = self.reward_model(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )
            reward = outputs.item()
        return reward

    def step(self, action):
        self.generated.append(int(action))
        done = len(self.generated) >= self.max_length or action == self.tokenizer.eos_token_id

        # Prepare the next observation
        obs = self.generated + [self.tokenizer.pad_token_id] * (self.max_length - len(self.generated))
        obs = np.array(obs[:self.max_length], dtype=np.int32)
        self.current_obs = obs

        reward = 0.0
        if done:
            text = self.tokenizer.decode(self.generated, skip_special_tokens=True)
            reward = self.compute_reward(text)

        return obs, reward, done, {}

class CustomMLPExtractor(nn.Module):
    """
    Custom MLP Extractor that simply passes through the features.
    """
    def __init__(self, latent_dim):
        super(CustomMLPExtractor, self).__init__()
        self.latent_dim = latent_dim

    def forward(self, features):
        return features, features  # Return both policy and value features

    def forward_actor(self, features):
        return features

    def forward_critic(self, features):
        return features

class CustomGPT2Policy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        # Set the features extractor to identity since GPT-2 handles feature extraction
        self.features_extractor = None  # We'll override the extract_features method
        super(CustomGPT2Policy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            **kwargs
        )

        self.gpt2 = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
        # self.gpt2.gradient_checkpointing_enable()
        # self.gpt2.resize_token_embeddings(len(tokenizer))  # Resize embeddings to 5000

        self.gpt2.config.use_cache = False  # Disable caching to save memory

        # Define action and value networks
        self.latent_dim = self.gpt2.config.hidden_size

        # Create a custom MLP extractor that passes through the features
        self.mlp_extractor = CustomMLPExtractor(self.latent_dim)

        # Action and value networks
        self.action_net = nn.Linear(self.latent_dim, self.action_space.n).to(device)
        self.value_net = nn.Linear(self.latent_dim, 1).to(device)

        # Initialize the distribution
        self.dist = CategoricalDistribution(self.action_space.n)

    def extract_features(self, obs):
        obs = obs.long().to(device)
        outputs = self.gpt2.transformer(obs)
        hidden_states = outputs.last_hidden_state  # Shape: [batch_size, seq_length, hidden_size]
        last_hidden = hidden_states[:, -1, :]  # Take the last token
        return last_hidden

    def forward(self, obs, deterministic=False):
        with autocast():
            features = self.extract_features(obs)
            latent_pi, latent_vf = self.mlp_extractor(features)

            # Policy
            action_logits = self.action_net(latent_pi)
            distribution = self.dist.proba_distribution(action_logits=action_logits)
            if deterministic:
                actions = torch.argmax(action_logits, dim=-1)
            else:
                actions = distribution.sample()

            log_probs = distribution.log_prob(actions)

            # Value function
            values = self.value_net(latent_vf).squeeze(-1)

            return actions, values, log_probs

    def _predict(self, observation: torch.Tensor, deterministic: bool = False):
        actions, _, _ = self.forward(observation, deterministic)
        return actions.cpu()

    def get_distribution(self, obs):
        features = self.extract_features(obs)
        latent_pi, _ = self.mlp_extractor(features)
        action_logits = self.action_net(latent_pi)
        distribution = self.dist.proba_distribution(action_logits=action_logits)
        return distribution

    def predict_values(self, obs):
        features = self.extract_features(obs)
        _, latent_vf = self.mlp_extractor(features)
        values = self.value_net(latent_vf).squeeze(-1)
        return values

    def evaluate_actions(self, obs, actions):
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)

        # Policy
        action_logits = self.action_net(latent_pi)
        distribution = self.dist.proba_distribution(action_logits=action_logits)
        log_probs = distribution.log_prob(actions.to(device))
        entropy = distribution.entropy()

        # Value function
        values = self.value_net(latent_vf).squeeze(-1)

        return log_probs, values, entropy

if __name__ == '__main__':
    # Create the environment
    env = LanguageModelEnv(tokenizer, reward_model, max_length=10)

    # Create the TRPO model
    model = TRPO(CustomGPT2Policy, env, verbose=1, device=device, batch_size=1, n_steps=16)

    # Train the model
    model.learn(total_timesteps=1)

    # Save the trained model
    model.save("gpt2_trpo_model")
    # tokenizer.save_pretrained("gpt2_trpo_model")
