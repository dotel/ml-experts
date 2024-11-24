import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch.nn as nn

# Define the reward model architecture
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

# Load tokenizer
base_model = 'lvwerra/gpt2-imdb'
tokenizer = GPT2Tokenizer.from_pretrained('./reward-model-imdb-dataset')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Load the reward model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
reward_model = GPT2RewardModel(model_name=base_model, tokenizer=tokenizer).to(device)
reward_model.load_state_dict(torch.load('./reward-model-imdb-dataset/reward-model-imdb-dataset.pth', map_location=device))

# Evaluation function
def evaluate_model(model, tokenizer, prompt, max_length=256):
    model.eval()
    with torch.no_grad():
        # Tokenize the prompt
        encoding = tokenizer(
            prompt,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
        
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        # Pass through the model
        reward_score = model(input_ids=input_ids, attention_mask=attention_mask, apply_sigmoid=True)
        
    return reward_score.item()

# Test with an example prompt
prompts = [
    "Zentropa has much in common with The Third Man, another noir-like film set among the rubble of postwar Europe. Like TTM, there is much inventive camera work. There is an innocent American who gets emotionally involved with a woman he doesn't really understand, and whose naivety is all the more striking in contrast with the natives.<br /><br />But I'd have to say that The Third Man has a more well-crafted storyline. Zentropa is a bit disjointed in this respect. Perhaps this is intentional: it is presented as a dream/nightmare, and making it too coherent would spoil the effect. <br /><br />This movie is unrelentingly grim--noir in more than one sense; one never sees the sun shine. Grim, but intriguing, and frightening.",
    "This movie was really good.",
    "I rented I AM CURIOUS-YELLOW from my video store because of all the controversy that surrounded it when it was first released in 1967. I also heard that at first it was seized by U.S. customs if it ever tried to enter this country, therefore being a fan of films considered controversial I really had to see this for myself.<br /><br />The plot is centered around a young Swedish drama student named Lena who wants to learn everything she can about life. In particular she wants to focus her attentions to making some sort of documentary on what the average Swede thought about certain political issues such as the Vietnam War and race issues in the United States. In between asking politicians and ordinary denizens of Stockholm about their opinions on politics, she has sex with her drama teacher, classmates, and married men.<br /><br />What kills me about I AM CURIOUS-YELLOW is that 40 years ago, this was considered pornographic. Really, the sex and nudity scenes are few and far between, even then it's not shot like some cheaply made porno. While my countrymen mind find it shocking, in reality sex and nudity are a major staple in Swedish cinema. Even Ingmar Bergman, arguably their answer to good old boy John Ford, had sex scenes in his films.<br /><br />I do commend the filmmakers for the fact that any sex shown in the film is shown for artistic purposes rather than just to shock people and make money to be shown in pornographic theaters in America. I AM CURIOUS-YELLOW is a good film for anyone wanting to study the meat and potatoes (no pun intended) of Swedish cinema. But really, this film doesn't have much of a plot.",
    "I liked, it's good",
    "The movie was amazing.",
    "I didn't like this movie at all.",
    "The acting was terrible.",
    "The plot was confusing and poorly executed.",
    "I am neutral about this movie.",
    "OK movie, not bad."
]
# Evaluate the prompt
for prompt in prompts:
    reward_score = evaluate_model(reward_model, tokenizer, prompt)
    print(f"Prompt: {prompt} Reward Score: {reward_score}\n")
