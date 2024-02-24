import torch.nn as nn
import math
import torch

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, num_heads: int, attn_dropout: float) -> None:

        '''
        Computes self attention by multiplying each token against every other token in a dot product manner resulting in a relationship of one token against every other tokens or vice versa showing impact of each token which helps to focus where it is needed improving the attention of the architecture.
        In this we use Self Attention as stated in the paper and compute it over multiple heads
        
        Args:
        d_model: Embedding dimension
        num_heads: Number of attention heads
        attn_dropout: Dropout probability for attention

        Returns:
        attn_out: Attention output value
        '''
        super().__init__()
        self.d_model = d_model # Embedding vector size
        self.num_heads = num_heads # Number of heads
        self.head_dim = d_model // num_heads # Dimension of vector seen by each head
        self.wq = nn.Linear(d_model, d_model, bias=False) # Wq -> Query 
        self.wk = nn.Linear(d_model, d_model, bias=False) # Wk -> Key 
        self.wv = nn.Linear(d_model, d_model, bias=False) # Wv -> Value
        self.wo = nn.Linear(d_model, d_model, bias=False) # Wo -> Output projection
        self.dropout = nn.Dropout(attn_dropout)

    def attention_calculation(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor):

        '''
        Function to calculate the attention by using Scaled Dot Product Attention

        attn(q,k,v) = softmax(q.k.T/sqrt(d_model)).V

        Args:
        query: query weights tensor
        key: key weights tensor
        value: value weights tensor
        mask: attention_mask to indicate if the calculation needs to be performed on a given token. For PAD token ignoring and autoregressive while cross attention
        dropout: attention dropout probability
        '''
        
        # (bs, num_heads, seq_len, head_dim) --> (bs, num_heads, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            # A very low (indicating -inf) to the positions where mask == 0 so that in softmax it becomes 0
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) # (bs, num_heads, seq_len, seq_len) # Apply softmax

        if self.dropout is not None:
            attention_scores = self.dropout(attention_scores)

        # (bs, h, seq_len, seq_len) --> (bs, h, seq_len, head_dim)
        return (attention_scores @ value), attention_scores

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: torch.Tensor):

        query = self.wq(q) # (bs, seq_len, d_model)
        key = self.wk(k) # (bs, seq_len, d_model)
        value = self.wv(v) # (bs, seq_len, d_model)

        # d_model = num_heads*head_dim
        # (bs, seq_len, d_model) --> (bs, seq_len, num_heads, head_dim) --> (bs, num_heads, seq_len, head_dim)
        query = query.view(query.shape[0], query.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.num_heads, self.head_dim).transpose(1, 2)

        # Calculate attention
        x, self.attention_scores = self.attention_calculation(query, key, value, attn_mask)
        
        # Concatenate all the heads together
        # (bs, num_heads, seq_len, head_dim) --> (bs, seq_len, num_heads, head_dim) --> (bs, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.d_model)

        # Learn all the features concatenated by output projection layer since all heads are concatenated
        # (bs, seq_len, d_model)  
        attn_out = self.wo(x)
        return attn_out