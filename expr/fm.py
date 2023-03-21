import pathlib

from pysat.examples.fm import FM
from pysat.formula import WCNF, CNF
import os


php_path = pathlib.Path("data/php/")
php_prob = sorted(php_path.glob(f"*.zip"), key=lambda f: int(f.name.split('/')[-1].split('_')[0]))


if __name__ == '__main__':
    with open('fm.csv', 'w') as f:
        f.write('cnf, n_vars, n_clauses, cost, solving time\n')
        for prob in php_prob:
            cnf = CNF(from_file=prob)
            wcnf = WCNF()
            for clause in cnf.clauses:
                wcnf.append(clause, weight=1)
            fm = FM(wcnf, verbose=1)
            fm.compute()
            line = f'{prob.name}, {wcnf.nv}, {len(wcnf.soft)}, {fm.cost}, {fm.oracle_time():.4f}'
            f.write(line + '\n')
            f.flush()
            print(line)
