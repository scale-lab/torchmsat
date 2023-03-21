from pysat.examples.genhard import CB, GT, PAR, PHP


def gen_php():
    for n_holes in range(1, 101):
        print(n_holes)
        cnf = PHP(n_holes)
        cnf.to_file(fname=f"data/php/{cnf.nv}_{len(cnf.clauses)}_{n_holes}.zip")


def gen_cb():
    for size in range(1, 101):
        print(size)
        cnf = CB(size)
        cnf.to_file(fname=f"data/cb/{cnf.nv}_{len(cnf.clauses)}_{size}.zip")


def gen_gt():
    for size in range(1, 101):
        print(size)
        cnf = GT(size)
        cnf.to_file(fname=f"data/gt/{cnf.nv}_{len(cnf.clauses)}_{size}.zip")


def gen_par():
    for size in range(1, 101):
        print(size)
        cnf = PAR(size)
        cnf.to_file(fname=f"data/par/{cnf.nv}_{len(cnf.clauses)}_{size}.zip")


if __name__ == "__main__":
    gen_php()
    gen_cb()
    gen_gt()
    gen_par()
