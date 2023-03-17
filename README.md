# torchmsat


## Installation

`pip install torchmsat`

## Usage

Clauses are in the [WDIMACS Input format](http://www.maxhs.org/docs/wdimacs.html).

```Python
from torchmsat import solver

nv = 2
clauses = [[1, 2],
           [1, -2],
           [-1, 2],
           [-1, -2]]

s = solver.Solver(prob.nv, prob.clauses)
cost, sol = s.compute()
```

Output:

```Python
(1, tensor([[-1.,  1.,  1., -1.,  1., -1.]]))
```

where `1` is the minimum number of unsatisfied clauses, and the tensor represents literal assignments `-1` for `0` and `1` for `1`.

