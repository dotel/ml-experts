import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import A2C
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class TextGenerationEnv(gym.Env):
    """
    A Gym environment for text generation.
    """
    def __init__(self, reward_model, tokenizer, max_steps=20):
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.max_steps = max_steps

        self.current_text = []
        self.current_step = 0

        # Initialize action and observation spaces
        self.vocab_size = len(tokenizer)
        self.action_space = spaces.Discrete(self.vocab_size)
        self.observation_space = spaces.Box(
            low=0, high=self.vocab_size - 1, shape=(self.max_steps,), dtype=np.int32
        )

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment and return the initial state.
        """
        super().reset(seed=seed)
        self.current_text = [self.tokenizer.bos_token_id]  # Start with BOS token
        self.current_step = 0
        return self._get_obs(), {}

    def step(self, action):
        """
        Take an action (append a token) and return the new state, reward, and done flag.
        """
        # Add the action (token) to the text
        self.current_text.append(action)
        self.current_step += 1

        # Compute reward using reward model
        text = self.tokenizer.decode(self.current_text)
        reward = self._compute_reward(text)

        # Check if the episode is done
        terminated = self.current_step >= self.max_steps
        truncated = False  # Set to True if you have a truncation condition

        return self._get_obs(), reward, terminated, truncated, {}

    def _compute_reward(self, text):
        """
        Compute the reward for the generated text using the reward model.
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            reward = self.reward_model(inputs.input_ids, attention_mask=inputs.attention_mask).item()
        return reward

    def _get_obs(self):
        """
        Get the current observation (tokenized text padded to max_steps).
        """
        obs = np.array(
            self.current_text + [self.tokenizer.pad_token_id] * (self.max_steps - len(self.current_text)),
            dtype=np.int32
        )
        return obs

    def render(self):
        """
        Print the current text.
        """
        print(self.tokenizer.decode(self.current_text))

# Load reward model
class GPT2RewardModel(torch.nn.Module):
    def __init__(self, model_name="gpt2"):
        super(GPT2RewardModel, self).__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained(model_name)
        self.regression_head = torch.nn.Linear(self.gpt2.config.n_embd, 1)

    def forward(self, input_ids, attention_mask=None):
        transformer_outputs = self.gpt2.transformer(input_ids, attention_mask=attention_mask)
        hidden_states = transformer_outputs.last_hidden_state
        pooled_output = hidden_states[:, -1, :]  # Use the last token's hidden state
        reward = self.regression_head(pooled_output).squeeze(-1)
        return reward


if __name__ == "__main__":
    # Load reward model
    reward_model = GPT2RewardModel()
    reward_model.load_state_dict(torch.load("./reward_model-2-epoc/reward_model.pth"))
    reward_model.eval()

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("./reward_model-2-epoc")
    tokenizer.pad_token = tokenizer.eos_token  # Ensure padding token is defined

    # Create the environment
    env = TextGenerationEnv(reward_model, tokenizer, max_steps=20)

    # Train using A2C
    model = A2C("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    model.save("a2c_text_generation_model")

    # Test the model
    obs, _ = env.reset()
    for _ in range(20):  # Generate a sequence
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        if done:
            break
        env.render()
