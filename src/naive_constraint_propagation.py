import time
import numpy as np
from problem_define import slitherlink
from generate_problem import *
from constraint_propagation_forward import *

"""
This method only tries to generate a solution based on the edges that must be selected, without backtracking, 
so there is a high probability that a valid solution will not be generated.
"""


def is_legal_solution(problem):
    """
    Determine whether the generated solution satisfies all constraints
    """
    # The row and column coordinates of the starting point and the row and column coordinates
    # of the end point of each edge in the solution
    start_row, start_col, end_row, end_col = [], [], [], []
    edge_count = 0  # Total number of selected edges
    for row in range(problem.nrow+1):
        for col in range(problem.ncol):
            if problem.row_solution[row,col] == 1:
                edge_count += 1
                start_row.append(row)
                start_col.append(col)
                end_row.append(row)
                end_col.append(col+1)
    for row in range(problem.nrow):
        for col in range(problem.ncol+1):
            if problem.col_solution[row,col] == 1:
                edge_count += 1
                start_row.append(row)
                start_col.append(col)
                end_row.append(row+1)
                end_col.append(col)

    # Digital constraints are not met
    num_sum = np.sum(problem.constraint=='1') + 2*np.sum(problem.constraint=='2') + 3*np.sum(problem.constraint=='3')
    if 2*edge_count < num_sum:
        return False

    first_pos = (start_row[0], start_col[0])
    next_pos = (end_row[0], end_col[0])
    can_select = [True] * edge_count
    can_select[0] = False  # The first edge has been selected
    count = 1  # Number of edges forming the first loop
    update = True
    while update:
        update = False
        for idx in range(edge_count):
            if can_select[idx] and start_row[idx] == next_pos[0] and start_col[idx] == next_pos[1]:
                next_pos = (end_row[idx], end_col[idx])
                count += 1
                can_select[idx] = False
                update = True
                break
            if can_select[idx] and end_row[idx] == next_pos[0] and end_col[idx] == next_pos[1]:
                next_pos = (start_row[idx], start_col[idx])
                count += 1
                can_select[idx] = False
                update = True
                break
        # If a loop is formed, then jump out of the while loop
        if next_pos[0]==first_pos[0] and next_pos[1] == first_pos[1]:
            break
    
    if count == edge_count:  # The generated loop is the only loop in the solution
        return True
    else:
        return False


def naive_constraint_propagation(problem):
    remain = problem.constraint.copy()
    row_force, col_force = find_force(remain)

    for row_sol_pos in row_force:
        problem.row_solution[row_sol_pos] = 1
    for col_sol_pos in col_force:
        problem.col_solution[col_sol_pos] = 1
    
    if is_legal_solution(problem):
        print('success')
        problem.print_solution()
    else:
        print('Either there is no solution')
        print('or the solution cannot be obtained simply from the way of constraint propagation')


if __name__ == '__main__':
    problem = generate_problem_from_url(url='http://www.puzzle-loop.com/?v=0&size=0')
    problem.print_problem()

    start_time = time.time()
    naive_constraint_propagation(problem)
    end_time = time.time()
    print()
    print('start time: {}'.format(start_time))
    print('end time: {}'.format(end_time))
    print('time cost: {}'.format(end_time-start_time))
