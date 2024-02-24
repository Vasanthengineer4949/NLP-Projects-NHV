import torch
import torch.nn as nn
from config import *

class TokenEmbeddings(nn.Module):
    
    def __init__(self, d_model: int, vocab_size: int) -> None:
        
        '''
        Computes the token embeddings for input tokens
        
        Args:
        d_model - Input representation(Embedding) dimension
        vocab_size - Size of vocabulary
        
        Returns:
        token_embedding - Embedding of Input Tokens
        '''
        
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(
                                        num_embeddings=vocab_size,
                                        embedding_dim=d_model
                                    ) # Shape: (vocab_size, embed_dim)
        
    def forward(self, x: torch.Tensor): # Input - (bs, seq_len) -> Output: bs, seq_len, d_model
        token_embedding = self.embedding(x) # Embedding created using Embedding class of Torch
        return token_embedding
    
class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model: int, seq_len: int, dropout_p: float) -> None:
        
        '''
        Computes the Positional Encoding for the Input Tokens positions
        
        Args:
        d_model - Embedding dimension
        seq_len - Number of tokens in the input sequence
        dropout - Dropout value
        
        Returns:
        pos_encoding - Encoded position values
        '''
        
        super().__init__()
        
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout_p = dropout_p
        
        self.dropout = nn.Dropout(self.dropout_p) # Dropout layer
        position_encodings = torch.zeros(self.seq_len, self.d_model) # Shape: (seq_len, d_model)
        positions = torch.arange(0, self.seq_len, dtype=torch.float).unsqueeze(1) # Shape: (seq_len) -> (seq_len, 1)
        even_odd_i = torch.arange(0, self.d_model, 2).float() # 2i # Shape: (d_model/2)
        div_freqs_term = torch.pow(10000, even_odd_i/d_model) # 10000**2i/dmodel
        position_encodings[:, 0::2] = torch.sin(positions*div_freqs_term) # Shape: (seq_len, d_model)
        position_encodings[:, 1::2] = torch.cos(positions*div_freqs_term) # Shape: (seq_len, d_model)
        position_encodings = position_encodings.unsqueeze(0) # Shape: (1, seq_len, d_model)
        self.register_buffer('position_encodings', position_encodings) # to be a part of module state but not a parameter of module
        
    def forward(self, x: torch.Tensor):
        x = x + (self.position_encodings[:, :x.shape[1], :]).requires_grad_(False) # Adding positional encodings
        pos_encoding = self.dropout(x) # Addding dropout to the position embedded input
        return pos_encoding
    
class InputEmbeddings(nn.Module):

    def __init__(self):

        '''
        An Embedding module which will compute the input embedding by adding the token and position embedding of the input tokens

        Returns:
        inp_embedding - Input Embedding
        '''
        super().__init__()


        self.d_model = D_MODEL
        self.vocab_size = VOCAB_SIZE
        self.seq_len = MAX_SEQ_LEN
        self.dropout_p = FF_DROPOUT
        self.token_embedding = TokenEmbeddings(self.d_model, self.vocab_size)
        self.positional_encoding = PositionalEncoding(self.d_model, self.seq_len, self.dropout_p)

    def forward(self, x: torch.Tensor):

        token_embed_x = self.token_embedding(x)
        inp_embedding = self.positional_encoding(token_embed_x)
        return inp_embedding
