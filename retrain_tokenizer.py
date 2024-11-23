# Install the tokenizers library if you haven't already
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
import os

from datasets import concatenate_datasets

ds = load_dataset("Anthropic/hh-rlhf")

full_dataset = concatenate_datasets([ds['train'], ds['test']])



# Create a directory to store the tokenizer files
tokenizer_dir = './tokenizer_500'
os.makedirs(tokenizer_dir, exist_ok=True)

# Collect all texts from your dataset
texts = [example['chosen'] for example in full_dataset] + [example['rejected'] for example in full_dataset]

# Save texts to a file (required for training)
with open('train_texts.txt', 'w', encoding='utf-8') as f:
    for text in texts:
        f.write(text + '\n')

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Train the tokenizer
tokenizer.train(files='train_texts.txt', vocab_size=100, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

# Save the tokenizer
tokenizer.save_model(tokenizer_dir)

from transformers import PreTrainedTokenizerFast

# Load the saved tokenizer
tokenizer = ByteLevelBPETokenizer(
    os.path.join(tokenizer_dir, "vocab.json"),
    os.path.join(tokenizer_dir, "merges.txt")
)

# Convert to PreTrainedTokenizerFast and save as tokenizer.json
fast_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    unk_token="<unk>",
    pad_token="<pad>",
    cls_token="<s>",
    sep_token="</s>",
    mask_token="<mask>",
)

# Save the tokenizer.json
fast_tokenizer.save_pretrained(tokenizer_dir)

