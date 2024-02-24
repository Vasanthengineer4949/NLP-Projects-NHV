import torch.nn as nn
import torch

class LayerNorm(nn.Module):

    def __init__(self, d_model: int, eps: float) -> None:

        '''
        Normalizes the values layerwise by the formula: X-mean/(std + eps)
        Here eps -> epsilon is a very small value preventing zero division.
        Here also two other parameters known as alpha and beta are introduced where alpha is multiplicative and beta is additive so that it introduces some deviations in the data creating some variety since by this normalization all values will fall under 0 to 1. 
        alpha and beta are learnable parameters

        Returns:
        layer_norm_out - Normalized layer value output
        '''
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x:torch.Tensor):
        mean = x.mean(dim=-1, keepdim=True) # Finding mean across the last dimension while retaining the dimension
        std = x.std(dim=-1, keepdim=True) # Finding mean across the last dimension while retaining the dimension
        x_normalized = (x - mean)/(std + self.eps) # X - mean / std + eps
        layer_norm_out = self.alpha * x_normalized + self.beta # Variance creation by alpha * x_normalized + beta
        return layer_norm_out

class ResidualConnection(nn.Module):

    def __init__(self, d_model: int, res_dropout: float):

        '''
        In the paper the Add and Layer Norm is handled by this Residual connection layer where the inputs is passed to a layer and that output is added with the input and is normalized at the layer level
        x = LayerNorm(x + sublayer(x))

        Args:
        res_dropout - residual dropout probability

        Returns:
        add_layer_norm_out  - Add and Layer Norm output
        '''

        super().__init__()
        self.dropout_layer = nn.Dropout(res_dropout)
        self.layer_norm_layer = LayerNorm(d_model, 1e-6)

    def forward(self, x, sublayer):

        sublayer_out = self.dropout_layer(sublayer(self.layer_norm_layer(x)))
        add_layer_norm_out = x + sublayer_out
        return add_layer_norm_out
        
