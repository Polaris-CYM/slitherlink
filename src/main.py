import time
from generate_problem import *
from backtracking import backtracking_solve
from backjumping import backjumping_solve
from constraint_propagation_forward import constraint_propagation_forward
from sat_solver import sat_solve
from minizinc_solve import minizinc_solve

if __name__ == '__main__':
    alltime = []
    for _ in range(10):
        # Get a random puzzle each time size=0--5x5 normal, size=4--5x5 hard, size=1--10x10 normal, size=10--7x7 normal
        problem = generate_problem_from_url(url='http://www.puzzle-loop.com/?v=0&size=0')

        start_time = time.time()
        backtracking_solve(problem)
        bt_time = time.time() - start_time

        start_time = time.time()
        backjumping_solve(problem)
        bj_time = time.time() - start_time

        start_time = time.time()
        constraint_propagation_forward(problem)
        cf_time = time.time() - start_time
        
        start_time = time.time()
        sat_solve(problem)
        sat_time = time.time() - start_time

        start_time = time.time()
        minizinc_solve(problem, "gecode")
        ge_time = time.time() - start_time

        start_time = time.time()
        minizinc_solve(problem, "chuffed")
        chuf_time = time.time() - start_time

        alltime.append([bt_time, bj_time, cf_time, sat_time, chuf_time])

        print("="*50)
        print(alltime)
        print("="*50)

