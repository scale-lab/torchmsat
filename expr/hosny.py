import pathlib

from pysat.formula import CNF

from torchmsat.solver import Solver

data_path = pathlib.Path("data/gt/")
probs = sorted(data_path.glob("*.zip"), key=lambda f: int(f.name.split("/")[-1].split("_")[0]))


if __name__ == "__main__":
    timer = None
    with open("results/hosny-cpu-gt-1e-2_3.csv", "w") as f:
        f.write(
            "cnf, n_vars, n_clauses, cost, build time, solving time, total search time, n_sols\n"
        )
        for prob in probs[:50]:
            print(prob.name)
            cnf = CNF(from_file=prob)
            s = Solver(cnf.nv, cnf.clauses, lr=1e-2)
            trace = s.compute(masked_steps=3000, unmasked_steps=3000)
            cost = trace["cost"]
            build_time = trace["nn_build_time"]
            solve_time = trace["max_sat_time"]
            total_time = trace["total_time"]
            n_sols = len(trace["sols"][cost])
            line = (
                f"{prob.name}, {cnf.nv}, {len(cnf.clauses)}, {cost}, "
                f"{build_time:.4f}, {solve_time:.4f}, {total_time:.4f}, {n_sols}"
            )
            f.write(line + "\n")
            f.flush()
            print(line)
            with open(f"results/hosny-cpu-gt-1e-2_3/{prob.name}.sol", "w") as sol_file:
                for sol in trace["sols"][cost]:
                    sol_file.write(f"{sol[0]}; {sol[1]}")
