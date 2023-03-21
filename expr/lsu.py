import pathlib
from threading import Timer

from pysat.examples.lsu import LSU
from pysat.formula import CNF, WCNF

data_path = pathlib.Path("data/gt/")
probs = sorted(data_path.glob("*.zip"), key=lambda f: int(f.name.split("/")[-1].split("_")[0]))


def interrupt(s):
    s.interrupt()


if __name__ == "__main__":
    timer = None
    with open("lsu-gt.csv", "w") as f:
        f.write("cnf, n_vars, n_clauses, cost, solving time\n")
        for prob in probs:
            print(prob.name)
            cnf = CNF(from_file=prob)
            wcnf = WCNF()
            for clause in cnf.clauses:
                wcnf.append(clause, weight=1)
            lsu = LSU(wcnf, verbose=0, expect_interrupt=True)
            if timer is not None:
                timer.cancel()
            timer = Timer(60, interrupt, [lsu])
            timer.start()
            lsu.solve()
            line = f"{prob.name}, {wcnf.nv}, {len(wcnf.soft)}, {lsu.cost}, {lsu.oracle_time():.4f}"
            f.write(line + "\n")
            f.flush()
            print(line)
