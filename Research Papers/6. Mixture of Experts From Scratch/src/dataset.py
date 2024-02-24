from datasets import Dataset as HFDataset
from datasets import load_dataset
from torch.utils.data import Dataset
from tokenizers import Tokenizer
import pandas as pd
import torch

class ClassificationDataset(Dataset):

    def __init__(self, dataset_id: str, tokenizer: Tokenizer, max_seq_len: int,  src_cln_name: str, tgt_cln_name: str):

        '''
        A Pytorch Dataset class to load the classification dataset from huggingface with all the preprocessing and tokenization completed.

        Args:
        dataset_id: Dataset ID on the datasets hub
        tokenizer: Whitespace Word Level Trained Tokenizer
        max_seq_len: Maximum sequence length for both src and tgt features
        src_cln: Source column name
        tgt_cln: Target column name

        Returns:
        model_inp: Model input data
        '''

        super().__init__()

        # self.dataset_id = dataset_id
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.src_cln_name = src_cln_name
        self.tgt_cln_name = tgt_cln_name
        self.dataset_id = dataset_id
        self.dataset = HFDataset.from_pandas(pd.read_parquet(dataset_id).sample(frac=1))
        # self.dataset = load_dataset(self.dataset_id, split="train")
        self.sos_token = torch.tensor([tokenizer.token_to_id("[SOS]")]).to(torch.int64)
        self.eos_token = torch.tensor([tokenizer.token_to_id("[EOS]")]).to(torch.int64)
        self.pad_token = torch.tensor([tokenizer.token_to_id("[PAD]")]).to(torch.int64)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):

        data = self.dataset[index]
        src_data = data[self.src_cln_name] # Fetching the source data
        tgt_data = data[self.tgt_cln_name] # Fetching the target data

        encoder_inp_tokens = torch.tensor(self.tokenizer.encode(src_data).ids).to(torch.int64) # Tokenizing the source data
        decoder_inp_tokens = torch.tensor(self.tokenizer.encode(str(tgt_data)).ids).to(torch.int64) # Tokenizing the target data

        if encoder_inp_tokens.size(0) > self.max_seq_len-2:
            encoder_inp_tokens = encoder_inp_tokens[:self.max_seq_len-2]

        encoder_num_padding_tokens = self.max_seq_len - len(encoder_inp_tokens) - 2 # Calculating the number of source padding tokens
        decoder_num_padding_tokens = 2 - len(decoder_inp_tokens) - 1 # Calculating the number of target padding tokens

        encoder_inp_pad_tokens = torch.tensor([self.pad_token] * encoder_num_padding_tokens).to(torch.int64) # Encoder input padding tokens
        decoder_inp_pad_tokens = torch.tensor([self.pad_token] * decoder_num_padding_tokens).to(torch.int64) # Decoder input padding tokens 
        label_pad_tokens = torch.tensor([self.pad_token] * decoder_num_padding_tokens).to(torch.int64) # Label padding tokens

        encoder_inp_tokens = torch.cat([self.sos_token, encoder_inp_tokens, self.eos_token, encoder_inp_pad_tokens], dim=0) # Encoder input tokens padded
        decoder_inpp_tokens = torch.cat([self.sos_token, decoder_inp_tokens, decoder_inp_pad_tokens], dim=0) # Decoder input tokens padded
        label_tokens = torch.cat([decoder_inp_tokens, self.eos_token, label_pad_tokens], dim=0) # Label tokens padded
        
        encoder_attn_mask = (encoder_inp_tokens != self.pad_token).unsqueeze(0).unsqueeze(0).to(torch.int64) # Encoder attention mask
        auto_regressive_mask = torch.triu(torch.ones((1, decoder_inpp_tokens.size(0), decoder_inpp_tokens.size(0))), diagonal=1).type(torch.int64)
        decoder_attn_mask = (decoder_inpp_tokens != self.pad_token).unsqueeze(0).to(torch.int64) & auto_regressive_mask==0 # Decoder autoregressive attention mask

        model_inp = {}
        model_inp["encoder_input_ids"] = encoder_inp_tokens
        model_inp["decoder_input_ids"] = decoder_inpp_tokens
        model_inp["encoder_attention_mask"] = encoder_attn_mask
        model_inp["decoder_attention_mask"] = decoder_attn_mask
        model_inp["labels"] = label_tokens
        model_inp["source"] = src_data
        model_inp["target"] = tgt_data

        return model_inp # Returning the model inputs
         




        
