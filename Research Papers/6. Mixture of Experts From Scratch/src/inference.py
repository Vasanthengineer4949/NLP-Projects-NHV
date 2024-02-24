from transformer import build_transformer_model
from tokenizer import Tokenizer
from train_utils import TrainerUtils
import torch
from config import *

train_utils = TrainerUtils()
def classify(sentence: str):
    # Define the device, tokenizers, and model
    device = "cpu"
    tokenizer = Tokenizer.from_file("tokenizer.json")
    model = build_transformer_model().to(device)

    # Load the pretrained weights
    state = torch.load("moe/epoch-4-checkpoint.pth")
    model.load_state_dict(state['model'])

    # if the sentence is a number use it as an index to the test set
    seq_len = MAX_SEQ_LEN

    # classify the sentence
    model.eval()
    with torch.no_grad():
        # Precompute the encoder output and reuse it for every generation step
        source = tokenizer.encode(sentence)
        source = torch.cat([
            torch.tensor([tokenizer.token_to_id('[SOS]')], dtype=torch.int64), 
            torch.tensor(source.ids, dtype=torch.int64),
            torch.tensor([tokenizer.token_to_id('[EOS]')], dtype=torch.int64),
            torch.tensor([tokenizer.token_to_id('[PAD]')] * (seq_len - len(source.ids) - 2), dtype=torch.int64)
        ], dim=0).to(device)
        source_mask = (source != tokenizer.token_to_id('[PAD]')).unsqueeze(0).unsqueeze(0).int().to(device)

        model_out = train_utils.generate(model=model, source=source, src_attn_mask=source_mask, tokenizer=tokenizer)
    return model_out

print(classify("i feel very good"))