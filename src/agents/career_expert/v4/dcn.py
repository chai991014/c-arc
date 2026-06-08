import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossNetworkV2(nn.Module):
    """Implementation of the DCN V2 Matrix Cross Layer"""

    def __init__(self, input_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.W = nn.ParameterList([nn.Parameter(torch.empty(input_dim, input_dim)) for _ in range(num_layers)])
        self.b = nn.ParameterList([nn.Parameter(torch.empty(input_dim)) for _ in range(num_layers)])

        for w, b in zip(self.W, self.b):
            nn.init.xavier_normal_(w)
            nn.init.zeros_(b)

    def forward(self, x0):
        xl = x0
        for i in range(self.num_layers):
            # DCN V2 Core Formula: x_{l+1} = x_0 * (W_l * x_l + b_l) + x_l
            xl_w = F.linear(xl, self.W[i], self.b[i])
            xl = x0 * xl_w + xl
        return xl


class DCNv2(nn.Module):
    """The Deep & Cross Network V2 Architecture"""

    def __init__(self, input_dim, embed_dim=256, cross_layers=2, deep_layers=[256, 128, 64]):
        super().__init__()

        # 1. Embedding Layer (Converts 31k sparse array into dense concepts)
        self.embedding = nn.Linear(input_dim, embed_dim)

        # 2. Parallel Deep and Cross Paths
        self.cross = CrossNetworkV2(embed_dim, cross_layers)

        layers = []
        in_dim = embed_dim
        for out_dim in deep_layers:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            in_dim = out_dim
        self.deep = nn.Sequential(*layers)

        # 3. Final Prediction Combiner
        self.fc_out = nn.Linear(embed_dim + deep_layers[-1], 1)

    def forward(self, x):
        # Step 1: Learn the dense embeddings of the provided skills
        x_emb = torch.relu(self.embedding(x))

        # Step 2: Pass embeddings into parallel concept networks
        x_cross = self.cross(x_emb)
        x_deep = self.deep(x_emb)

        # Step 3: Combine and predict final readiness capability
        out = torch.cat([x_cross, x_deep], dim=1)
        score = self.fc_out(out)
        return score.squeeze()
