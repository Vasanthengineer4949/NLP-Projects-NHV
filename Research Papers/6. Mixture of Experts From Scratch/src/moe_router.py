import torch
import torch.nn as nn
import torch.nn.functional as F

class NoisyTopkRouter(nn.Module):
    def __init__(self, d_model, num_experts, top_k):
        super(NoisyTopkRouter, self).__init__()
        
        self.top_k = top_k
        self.topkroute_linear = nn.Linear(d_model, num_experts)
        self.noise_linear =nn.Linear(d_model, num_experts)

    
    def forward(self, mh_output):

        logits = self.topkroute_linear(mh_output)
        noise_logits = self.noise_linear(mh_output)

        noise = torch.randn_like(logits)*F.softplus(noise_logits)
        noisy_logits = logits + noise

        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, indices