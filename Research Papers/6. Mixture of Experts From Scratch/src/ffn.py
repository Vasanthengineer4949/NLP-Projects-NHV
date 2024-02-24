import torch.nn as nn
import torch

class FeedForward(nn.Module):

    def __init__(self, d_model: int, ff_dropout: float) -> None:

        '''
        A FeedForward Block with two linear layers along with a drouput in it. Here the dimensions would be involving d_model and d_ff which according to the paper is four times the d_model and is the inner dimension
        
        Args:
        d_model - Embedding dimension
        ff_dropout - Dropout value
        
        Returns:
        ff_out - Feed Forward Block Output'''
        super().__init__()
        self.d_model = d_model
        self.ff_dropout = ff_dropout
        self.d_ff = self.d_model * 4
        self.linear_1 = nn.Linear(self.d_model, self.d_ff) # Linear layer 1 
        self.linear_2 = nn.Linear(self.d_ff, self.d_model) # Linear Layer 2
        self.ff_dropout_layer = nn.Dropout(self.ff_dropout) # Drouput layer with p = ff_dropout

    def forward(self, x:torch.Tensor):
        linear_1_out = self.linear_1(x) # Linear layer 1 out # Shape: bs,seq_len, d_ff
        linear_1_out = torch.relu(linear_1_out) # Applying ReLU activation function
        linear_1_out = self.ff_dropout_layer(linear_1_out) # Applying Dropout
        ff_out = self.linear_2(linear_1_out) # Linear layer 2 -> FF Block output # Shape: bs,seq_len, d_model
        return ff_out

