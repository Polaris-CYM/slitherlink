import time
from generate_problem import *
from backtracking import backtracking_solve
from backjumping import backjumping_solve
from constraint_propagation_forward import constraint_propagation_forward
from sat_solver import sat_solve, is_legal_solution
from minizinc_solve import minizinc_solve

if __name__ == '__main__':
    back, jump, fore, sat, ge, ch = 0, 0, 0, 0, 0, 0

    for i in range(10):
        problem = generate_problem_from_url(url='http://www.puzzle-loop.com/?v=0&size=0')

        print('backtracking', i)
        if is_legal_solution(backtracking_solve(problem)):
            back += 1

        print('backjumping', i)
        if is_legal_solution(backjumping_solve(problem)):
            jump += 1

        print('forward checking', i)
        if is_legal_solution(constraint_propagation_forward(problem)):
            fore += 1

        print('sat', i)
        if is_legal_solution(sat_solve(problem)):
            sat += 1
        
        print('gecode Minizinc', i)
        if is_legal_solution(minizinc_solve(problem, "gecode")):
            ge += 1

        print('chuffed Minizinc', i)
        if is_legal_solution(minizinc_solve(problem, "chuffed")):
            ch += 1

    print(back, jump, fore, sat, ge, ch)
        

