from config import *
from embedding import InputEmbeddings
from encoder import Encoder
from decoder import Decoder
import torch.nn as nn

class Transformer(nn.Module):

    def __init__(self):

        '''
        A Transformer model created from scratch
        
        Returns:
        transformer_out - Transformer model output
        '''

        super().__init__()

        self.d_model = D_MODEL
        self.num_layers = NUM_LAYERS
        self.vocab_size = VOCAB_SIZE
        self.inp_embedding = InputEmbeddings()
        self.encoder = Encoder(self.d_model, self.num_layers)
        self.decoder = Decoder(self.d_model, self.num_layers)
        self.projection = nn.Linear(self.d_model, self.vocab_size)

    def forward(self, src_x, src_attn_mask, tgt_x, tgt_attn_mask):

        src_embed = self.inp_embedding(src_x)
        tgt_embed = self.inp_embedding(tgt_x)
        encoder_out = self.encoder(src_embed, src_attn_mask)
        decoder_out = self.decoder(tgt_embed, encoder_out, src_attn_mask, tgt_attn_mask)
        transformer_out = self.projection(decoder_out)
        return transformer_out
    
def build_transformer_model():
    transformer = Transformer()
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer

transformer = build_transformer_model()
