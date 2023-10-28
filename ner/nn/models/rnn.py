# AUTO-GENERATED (DO NOT MODIFY)
# NET IDS: YR269,VMF24

import logging
from dataclasses import dataclass
from typing import List

import torch
from torch import nn

from ner.nn.module import Module


class RNN(Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 1,
        bias: bool = True,
        nonlinearity: str = "tanh",
    ):
        """Documentation: https://pages.github.coecis.cornell.edu/cs4740/hw2-fa23/ner.nn.models.rnn.html."""
        super().__init__()

        assert num_layers > 0

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        logging.info(f"no shared weights across layers")

        nonlinearity_dict = {"tanh": nn.Tanh(), "relu": nn.ReLU(), "prelu": nn.PReLU()}
        if nonlinearity not in nonlinearity_dict:
            raise ValueError(f"{nonlinearity} not supported, choose one of: [tanh, relu, prelu]")
        self.nonlinear = nonlinearity_dict[nonlinearity]

        # TODO-5-1

        self.W = nn.ModuleList([nn.Linear(embedding_dim, hidden_dim) for i in range(num_layers)])


        self.U = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(num_layers)])

        self.V = nn.Linear(hidden_dim, output_dim)

        self.apply(self.init_weights)

    def _initial_hidden_states(
        self, batch_size: int, init_zeros: bool = False, device: torch.device = torch.device("cpu")
    ) -> List[torch.Tensor]:
        if init_zeros:
            hidden_states = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        else:
            hidden_states = nn.init.xavier_normal_(
                torch.empty(self.num_layers, batch_size, self.hidden_dim, device=device)
            )
        return list(map(torch.squeeze, hidden_states))

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Documentation: https://pages.github.coecis.cornell.edu/cs4740/hw2-fa23/ner.nn.models.rnn.html."""
        # TODO-5-2
        Z = self._initial_hidden_states(batch_size=embeddings.size(dim=0), device=embeddings.device)
        V_list = []
        for i in range(embeddings.size(dim=1)):

            for j in range(self.num_layers):
                Wt = self.W[j]
                W = Wt(embeddings[:, i, :])
                Ut = self.U[j]
                U = Ut(Z[j])

                Z[j] = self.nonlinear(W+U)

            V = self.V(Z[-1]) # (batch_size, output_dim)
            V_list.append(V)

        V_list = torch.stack(V_list, dim=1)
        return V_list
