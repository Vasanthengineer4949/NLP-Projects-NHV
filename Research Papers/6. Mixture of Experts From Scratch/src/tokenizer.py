import torch
from config import *
import torch.nn as nn
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from os import path

class TransformerTokenizerTrainer:

    def __init__(self, data_id, tokenizer_path):
        self.data_id = data_id
        self.tokenizer_path = tokenizer_path
        self.vocab_size = VOCAB_SIZE

    def dataset_load(self):
        dataset = load_dataset(self.data_id, split="train")
        return dataset
    
    def dataset_iterator(self):
        dataset = self.dataset_load()
        for data in dataset:
            yield data["text"]

    def build_tokenizer(self):
        if path.exists(self.tokenizer_path):
            tokenizer = Tokenizer.from_file(self.tokenizer_path)
        else:
            tokenizer = Tokenizer(WordLevel(unk_token="[UNK]")) # Instantiating a word level tokenizer that splits the sentences to give tokens at word levels with the unknown token as [UNK]
            tokenizer.pre_tokenizer = Whitespace() # Setting the split to be based on whitespace
            tokenizer_trainer = WordLevelTrainer(vocab_size=self.vocab_size, special_tokens=["[PAD]", "[SOS]", "[EOS]", "[UNK]"], min_frequency=2)
            tokenizer.train_from_iterator(self.dataset_iterator(), trainer=tokenizer_trainer)
            tokenizer.save(self.tokenizer_path)
        return tokenizer
    
transformer_tokenizer_trainer = TransformerTokenizerTrainer("C:/Users/vasan/.cache/huggingface/datasets/imdb", "tokenizer.json")
tokenizer = transformer_tokenizer_trainer.build_tokenizer()
