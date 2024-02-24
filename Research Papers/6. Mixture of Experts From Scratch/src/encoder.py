from config import *
from attention import MultiHeadAttention
from moe import SparseMoE
from layer_norm import ResidualConnection, LayerNorm
import torch.nn as nn
import torch

class EncoderBlock(nn.Module):

    def __init__(self):

        '''
        A Single Encoder block that can be stacked multiple times to form the encoder which uses two main blocks: Self Attention  Feed Forward connected with some Residual Connections

        Returns:
        enc_block_out - Encoder Block Output
        '''

        super().__init__()

        self.d_model = D_MODEL # Embedding dimension
        self.num_heads = NUM_HEADS # Number of heads
        self.attn_dropout = ATTN_DROPOUT # Attention dropout value
        self.ff_dropout = FF_DROPOUT # Feed forward dropout value
        self.res_dropout = RES_DROPOUT # Residual connection dropout value
        self.num_experts = NUM_EXPERTS # Number of experts
        self.top_k = TOP_K # Number of Training experts
        self.self_attention = MultiHeadAttention(self.d_model, self.num_heads, self.attn_dropout) # Attention layer
        self.moe = SparseMoE(self.d_model, self.num_experts, self.top_k, self.ff_dropout) # MOE layer
        self.res_connection1 = ResidualConnection(self.d_model, self.res_dropout) # Residual connection layer1
        self.res_connection2 = ResidualConnection(self.d_model, self.res_dropout) # Residual connection layer2

    def forward(self, x: torch.Tensor, src_attn_mask: torch.Tensor):
        attn_out = self.res_connection1(x, lambda x: self.self_attention(x, x, x, src_attn_mask)) # Attention and Normalized output
        enc_block_out = self.res_connection2(attn_out, self.moe) # Feed Forward output normalized and produces encoder output
        return enc_block_out
    
class Encoder(nn.Module):

    def __init__(self, d_model: int, num_layers: int):

        '''
        Transformer encoder which has a stack of encoder blocks joined together
        
        Args:
        d_model: Embedding_dimension
        num_layers: Number of layers in the stack

        Returns:
        encoder_out: Encoder output
        '''
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock() for _ in range(num_layers)])
        self.layer_norm = LayerNorm(d_model, EPS)

    def forward(self, x:torch.Tensor, attn_mask: torch.Tensor):
        
        for layer in self.layers:
            x = layer(x, attn_mask)
        encoder_out = self.layer_norm(x)
        return encoder_out
