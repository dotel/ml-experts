import gym
from gym import spaces
import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from torch.nn import functional as F
import torch.nn as nn
from transformers import AutoConfig, GPT2LMHeadModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

class GPT2RewardModel(nn.Module):
    def __init__(self):
        super(GPT2RewardModel, self).__init__()
        config = AutoConfig.from_pretrained("distilgpt2")
        config.n_layer = 6
        config.n_embd = 768  # Match hidden size with the saved checkpoint
        config.n_head = 12

        self.gpt2 = GPT2LMHeadModel(config).to(device)
        
        # self.gpt2.gradient_checkpointing_enable()
        self.dropout = nn.Dropout(0.1)
        self.regression_head = nn.Linear(config.n_embd, 1)

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

class TextGenEnv(gym.Env):
    def __init__(self, tokenizer, reward_model, max_length=50):
        super(TextGenEnv, self).__init__()
        self.tokenizer = tokenizer
        self.reward_model = reward_model
        self.max_length = max_length

        self.action_space = spaces.Discrete(len(tokenizer))
        self.observation_space = spaces.Box(
            low=0, high=len(tokenizer) - 1, shape=(self.max_length,), dtype=np.int32
        )

        self.current_sequence = []
        self.done = False

    def reset(self):
        self.current_sequence = []
        self.done = False
        return self._get_observation()

    def step(self, action):
        self.current_sequence.append(action)
        if len(self.current_sequence) >= self.max_length or action == self.tokenizer.eos_token_id:
            self.done = True

        observation = self._get_observation()
        reward = 0.0
        info = {}

        if self.done:
            text = self.tokenizer.decode(self.current_sequence)
            with torch.no_grad():
                inputs = self.tokenizer(
                    text,
                    return_tensors='pt',
                    truncation=True,
                    max_length=256,
                    padding='max_length',
                )
                reward = self.reward_model(
                    inputs['input_ids'], attention_mask=inputs['attention_mask']
                ).item()

        return observation, reward, self.done, info

    def _get_observation(self):
        obs = np.zeros(self.max_length, dtype=np.int32)
        obs[: len(self.current_sequence)] = self.current_sequence
        return obs

class CustomPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super(CustomPolicy, self).__init__(
            observation_space, action_space, lr_schedule, **kwargs
        )
        self.lm_head = GPT2LMHeadModel.from_pretrained('distilgpt2')
        self.tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
        self._build(lr_schedule)

    def _build(self, lr_schedule):
        # Override the actor and critic networks with the language model
        self.actor = self.lm_head
        self.critic = torch.nn.Linear(self.lm_head.config.n_embd, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_schedule(1))

    def forward(self, obs, deterministic=False):
        input_ids = torch.tensor(obs, dtype=torch.long).to(self.device)
        outputs = self.actor(input_ids)
        logits = outputs.logits[:, -1, :]  # Take the last token's logits
        action_probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(action_probs)
        if deterministic:
            actions = torch.argmax(action_probs, dim=-1)
        else:
            actions = dist.sample()
        log_probs = dist.log_prob(actions)
        values = self.critic(outputs.last_hidden_state[:, -1, :])
        return actions, values, log_probs

# Initialize tokenizer and reward model
tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
reward_model = GPT2RewardModel()
reward_model.load_state_dict(torch.load('./reward_model.pth'))
reward_model.eval()

# Create the environment
env = TextGenEnv(tokenizer, reward_model)

# Create the PPO model with the custom policy
model = PPO(
    CustomPolicy,
    env,
    verbose=1,
    device='cuda'  # or 'cpu'
)

# Train the model
model.learn(total_timesteps=10000)
