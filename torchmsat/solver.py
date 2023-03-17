import signal
from typing import List, Tuple

import torch


class _Model_(torch.nn.Module):
    def __init__(self, nv, clauses) -> None:
        super(_Model_, self).__init__()

        self.e = torch.ones((1, nv))
        x = torch.rand((1, nv))
        self.x = torch.nn.Parameter(x)

        self.W = torch.zeros((len(clauses), nv))

        self.target = torch.zeros((1, len(clauses)))

        self.SAT = torch.zeros((1, len(clauses)))
        for i, clause in enumerate(clauses):
            for literal in clause:
                value = 1.0 if literal > 0 else -1.0
                literal_idx = abs(literal) - 1
                self.W[i, literal_idx] = value
            self.SAT[0, i] = -len(clause)

        # Auxiliary for reporting a solution
        self.sol = torch.zeros_like(self.x)

    def forward(self):
        act = torch.tanh(self.e * self.x) @ self.W.T
        self.sol[self.x > 0] = 1.0
        self.sol[self.x <= 0] = -1.0
        return act

    def sat(self):
        unsat_clauses = (self.sol @ self.W.T) == self.SAT
        cost = torch.sum(unsat_clauses).item()
        return cost

    def __str__(self) -> str:
        return f'W={self.W}'


class Solver():
    def __init__(self, nv, clauses) -> None:
        signal.signal(signal.SIGINT, self.signal_handler)

        self.trace = {
            'start_time': 0.0,
            'nn_build_time': 0.0,
            'max_sat_time': 0.0,
            'nv': nv,
            'nc': len(clauses)
        }
        self.sols: List[Tuple] = []

        self.model = _Model_(nv, clauses)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-2)
        self.loss = torch.nn.MSELoss()

    def compute(self):
        for i in range(1000):
            self.optimizer.zero_grad()
            out = self.model()
            output = self.loss(out, self.model.target)
            output.backward()
            self.optimizer.step()

            self.sols.append((self.model.sat(), self.model.sol))

        return self.max_sat()

    def max_sat(self):
        max_sat = min(self.sols, key=lambda sol: sol[0])
        return max_sat  # returns (cost, assignment)

    def signal_handler(self, sig, frame):
        print(self.max_sat())
