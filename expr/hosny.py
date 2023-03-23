import pathlib

from pysat.formula import CNF

from torchmsat.solver import Solver

data_path = pathlib.Path("data/gt/")
probs = sorted(data_path.glob("*.zip"), key=lambda f: int(f.name.split("/")[-1].split("_")[0]))


if __name__ == "__main__":
    timer = None
    with open("hosny-gpu-gt.csv", "w") as f:
        f.write("cnf, n_vars, n_clauses, cost, build time, solving time, total search time\n")
        for prob in probs:
            print(prob.name)
            cnf = CNF(from_file=prob)
            s = Solver(cnf.nv, cnf.clauses)
            trace = s.compute(steps=5000)
            cost = trace["cost"]
            build_time = trace["nn_build_time"]
            solve_time = trace["max_sat_time"]
            total_time = trace["total_time"]
            line = (
                f"{prob.name}, {cnf.nv}, {len(cnf.clauses)}, {cost}, "
                f"{build_time:.4f}, {solve_time:.4f}, {total_time:.4f}"
            )
            f.write(line + "\n")
            f.flush()
            print(line)
