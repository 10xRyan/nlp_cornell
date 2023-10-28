from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ner.nn.module import Module


class FFNN(Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 1) -> None:
        super().__init__()

        assert num_layers > 0

        self.W = nn.Linear(embedding_dim, hidden_dim)
        self.linears = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(num_layers - 1)])
        self.V = nn.Linear(hidden_dim, output_dim)



        self.apply(self.init_weights)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        Z = self.W(embeddings)
        Z = nn.functional.relu(Z)

        for layer in self.linears:
          Z = nn.functional.relu(layer(Z))

        Y = self.V(Z)
        
        return Y
