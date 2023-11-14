import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random


from pysat.examples.genhard import PHP
cnf = PHP(10)

print(f"nv={cnf.nv}; nc={len(cnf.clauses)}")

# Constructing the network
key = random.PRNGKey(1)
scale = 1e-2

unmasked_steps = 10000
masked_steps = 1000
step_size = 1

e = jnp.ones((1, cnf.nv))
x = scale * random.normal(key, (1, cnf.nv))

_W = np.zeros((cnf.nv, len(cnf.clauses)))
_SAT = np.zeros((1, len(cnf.clauses)))
for clause_idx, clause in enumerate(cnf.clauses):
    for literal in clause:
        value = 1.0 if literal > 0 else -1.0
        literal_idx = abs(literal) - 1
        _W[literal_idx, clause_idx] = value
    _SAT[0, clause_idx] = -len(clause)
W = jnp.array(_W)
SAT = jnp.array(_SAT)

target = jnp.zeros((1, len(cnf.clauses)))

def forward(e, W, x):
    return jnp.tanh(jnp.dot(e * x, W))

def sol(x):
    return jnp.where(x > 0, 1., -1.)

def cost(W, SAT, sol):
    unsat_clauses = jnp.dot(sol, W) == SAT
    return jnp.sum(unsat_clauses)

def loss(e, W, x, target):
    act = forward(e, W, x)
    return -jnp.mean((act - target)**2)

@jit
def update(e, W, x, target):
    dx = grad(loss)(e, W, x, target)
    return x - step_size * dx

best_sol = sol(x)
best_cost = cost(W, SAT, best_sol)

print(f"sol={best_sol}; cost={best_cost}")

for i in range(unmasked_steps):
    x_after = update(e, W, x, target)
    # print(x, x_after)
    x = x_after
    cur_sol = sol(x)
    cur_cost = cost(W, SAT, cur_sol)
    if cur_cost < best_cost:
        best_cost = cur_cost
        best_sol = cur_sol

print(f"sol={best_sol}; cost={best_cost}")