import time

import torch


class _Model_(torch.nn.Module):
    def __init__(self, nv, clauses, use_gpu=False) -> None:
        super(_Model_, self).__init__()
        device = (
            torch.device("cuda") if torch.cuda.is_available() and use_gpu else torch.device("cpu")
        )

        self.e = torch.ones((1, nv), device=device)
        x = torch.rand((1, nv), device=device)
        self.x = torch.nn.Parameter(x)

        self.W = torch.zeros((nv, len(clauses)))

        self.target = torch.zeros((1, len(clauses)), device=device)

        self.SAT = torch.zeros((1, len(clauses)))
        for clause_idx, clause in enumerate(clauses):
            for literal in clause:
                value = 1.0 if literal > 0 else -1.0
                literal_idx = abs(literal) - 1
                self.W[literal_idx, clause_idx] = value
            self.SAT[0, clause_idx] = -len(clause)

        # Auxiliary for reporting a solution
        self.sol = torch.empty_like(self.x, device=device)

        # GPU
        self.W = self.W.to(device)
        self.SAT = self.SAT.to(device)

    def forward(self):
        act = torch.tanh(self.e * self.x) @ self.W
        self.sol[self.x > 0] = 1.0
        self.sol[self.x <= 0] = -1.0
        return act

    def sat(self):
        unsat_clauses = (self.sol @ self.W) == self.SAT
        cost = torch.sum(unsat_clauses).item()
        return cost

    def __str__(self) -> str:
        return f"W={self.W}"


class Solver:
    def __init__(self, nv, clauses, lr=1e-4, use_gpu=False) -> None:
        self.trace = {
            "start_time": 0.0,
            "nn_build_time": 0.0,
            "max_sat_time": 0.0,
            "total_time": 0.0,
            "cost": len(clauses),
            "nv": nv,
            "nc": len(clauses),
            "sols": {},
        }
        self.trace["start_time"] = time.time()

        self.model = _Model_(nv, clauses, use_gpu)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss = torch.nn.MSELoss()

        self.trace["nn_build_time"] = time.time() - self.trace["start_time"]

    def compute(self, steps=1000):
        solve_start_time = time.time()
        for i in range(steps):
            self.optimizer.zero_grad()
            out = self.model()
            output = self.loss(out, self.model.target)
            output.backward()
            self.optimizer.step()

            cost = self.model.sat()
            if cost < self.trace["cost"]:
                self.trace["cost"] = cost
                self.trace["max_sat_time"] = time.time() - solve_start_time
                self.trace["sols"][cost] = [(i, self.sol_str())]
            elif cost == self.trace["cost"] and self.sol_str() not in list(
                map(lambda sol: sol[1], self.trace["sols"][cost])
            ):
                # Pareto frontier solutions at the lowest cost so far (till this iteration)
                self.trace["sols"][cost].append((i, self.sol_str()))
        self.trace["total_time"] = time.time() - solve_start_time
        return self.max_sat()

    def max_sat(self):
        return self.trace

    def sol_str(self):
        return ",".join([str(1 if var > 0 else 0) for var in self.model.sol.flatten().tolist()]) + "\n"
