import torch
import torch.nn as nn
import torch.nn.functional as F
from moe_router import NoisyTopkRouter
from ffn import FeedForward
from config import *

class SparseMoE(nn.Module):
    def __init__(self, d_model, num_experts, top_k, ff_dropout):
        super(SparseMoE, self).__init__()
        self.router = NoisyTopkRouter(d_model, num_experts, top_k)
        self.experts = nn.ModuleList([FeedForward(d_model, ff_dropout) for _ in range(num_experts)])
        self.top_k = top_k

    def forward(self, x):
        gating_output, indices = self.router(x)
        final_output = torch.zeros_like(x)

        flat_x = x.view(-1, x.size(-1))
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))

        for i, expert in enumerate(self.experts):

            expert_mask = (indices == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)

            if flat_mask.any():
                expert_input = flat_x[flat_mask]
                expert_output = expert(expert_input)

                gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)
                weighted_output = expert_output * gating_scores

                final_output[expert_mask] += weighted_output.squeeze(1)

        return final_output