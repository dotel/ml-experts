tokenizer_dir="tokenizer_5k"

# Load the tokenizer
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=os.path.join(tokenizer_dir, 'tokenizer.json'),
    unk_token="<unk>",
    pad_token="<pad>",
    cls_token="<s>",
    sep_token="</s>",
    mask_token="<mask>",
)
tokenizer.pad_token = tokenizer.unk_token  # Set padding token

