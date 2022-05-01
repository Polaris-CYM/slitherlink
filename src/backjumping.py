import time
import numpy as np
# from pyparsing import col
from problem_define import slitherlink
from generate_problem import *
from backtracking import *


def row_solution_pos_legal(pos, nrow, ncol):
    """
    The positions of the horizontal lines should within [0 ... nrow, 0 ... ncol-1]
    """
    if 0 <= pos[0] <= nrow and 0<= pos[1] <= ncol - 1:
        return True
    else:
        return False


def col_solution_pos_legal(pos, nrow, ncol):
    """
    The positions of the vertical lines should within [0 ... nrow-1, 0 ... ncol]
    """
    if 0 <= pos[0] <= nrow - 1 and 0<= pos[1] <= ncol:
        return True
    else:
        return False


def find_forbid(remain):
    """
    Find the list of edges that are prohibited from appearing in the final solution
    """
    nrow, ncol = remain.shape
    row_forbid = []  # row_forbid = [(r1,c1), (r2,c2), ...] indicates that problem.row_solution(ri,ci) cannot be part of the solution
    col_forbid = []

    # situation 1: digit '1' is at the corner
    if remain[0,0] == '1':
        row_forbid.append((0,0))
        col_forbid.append((0,0))
    if remain[0,ncol-1] == '1':
        row_forbid.append((0,ncol-1))
        col_forbid.append((0,ncol))
    if remain[nrow-1,0] == '1':
        row_forbid.append((nrow,0))
        col_forbid.append((nrow-1,0))
    if remain[nrow-1,ncol-1] == '1':
        row_forbid.append((nrow,ncol-1))
        col_forbid.append((nrow-1,ncol))

    # situation 2: two '3' are connected
    for row in range(nrow):
        for col in range(ncol-1):
            if remain[row, col] == '3' and remain[row, col+1] == '3':
                if col_solution_pos_legal((row-1, col+1), nrow, ncol):
                    col_forbid.append((row-1, col+1))
                if col_solution_pos_legal((row+1, col+1), nrow, ncol):
                    col_forbid.append((row+1, col+1))
    for row in range(nrow-1): 
        for col in range(ncol):
            if remain[row,col] == '3' and remain[row+1,col] == '3':
                if row_solution_pos_legal((row+1,col-1), nrow, ncol):
                    row_forbid.append((row+1,col-1))
                if row_solution_pos_legal((row+1,col+1), nrow, ncol):
                    row_forbid.append((row+1,col+1))
    
    return row_forbid, col_forbid


def forbid_allow(current_pos, next_dir, row_forbid, col_forbid):
    """
    Determine whether the next passed edge is in the forbidden list
    """
    if next_dir == 'right':
        return False if current_pos in row_forbid else True
    if next_dir == 'left':
        return False if (current_pos[0],current_pos[1]-1) in row_forbid else True
    if next_dir == 'down':
        return False if current_pos in col_forbid else True
    if next_dir == 'up':
        return False if (current_pos[0]-1,current_pos[1]) in col_forbid else True


def construct_jump_solution(problem, point_path):
    """
    Construct the solution of the puzzle based on point_path
    """
    problem.row_solution = np.zeros(shape=(problem.nrow+1, problem.ncol))
    problem.col_solution = np.zeros(shape=(problem.nrow, problem.ncol+1))

    path_len = len(point_path)
    for i in range(path_len - 1):
        start_point = point_path[i]
        end_point = point_path[i+1]
        dir = judge_direction(start_point, end_point)

        if dir == 'right':
            problem.row_solution[start_point] = 1
        elif dir == 'left':
            problem.row_solution[end_point] = 1
        elif dir == 'down':
            problem.col_solution[start_point] = 1
        elif dir == 'up':
            problem.col_solution[end_point] = 1
    print()
    print('BACKJUMPING SOLUTION:')
    problem.print_solution()
    return True


def backjumping_solve(problem):
    """
    use the backtracking search algorithm to solve the problem
    """
    nrow, ncol = problem.nrow, problem.ncol
    start_row, start_col = find_start(problem)  # Find the start point (function find_start is in backtracking.py)
    point_path = [(start_row, start_col)]  # point_path array: store the point path of the final solution
    dir_path = []  # Direction from the start point
    remain = problem.constraint.copy()  # Maintain a copy of the constraint, which can be increased or decreased at any time
    row_forbid, col_forbid = find_forbid(remain)
    current_pos = (start_row, start_col)  # Position of current node

    first_right = False  # whether the first step is to the right

    # Go down first
    next_dir = 'down'
    if (not remain_allow(remain, current_pos, next_dir)) or (not forbid_allow(current_pos, next_dir, row_forbid, col_forbid)):
        next_dir = 'right'
        first_right = True
    
    pos_a, pos_b = cal_pos_ab(current_pos, next_dir)
    if digit_pos_legal(pos_a, nrow, ncol):
        remain[pos_a] = minus_one(remain[pos_a])
    if digit_pos_legal(pos_b, nrow, ncol):
        remain[pos_b] = minus_one(remain[pos_b])

    next_pos = cal_next_pos(current_pos, next_dir)
    point_path.append(next_pos)
    dir_path.append(next_dir)
    current_pos = next_pos

    # type=1: initially go down/right, or reach a node for the first time
    # type=2: passed the node before, but the path eventually failed. Needed to try a different direction
    backtrack_type = 1
    initial = True

    while True:
        # If it goes back to the initial point, then the first step should not go down
        # Modify the following relevant information to restart the loop
        if initial == False and len(point_path) == 1:
            point_path = [(start_row, start_col)]
            dir_path = []
            remain = problem.constraint.copy()
            current_pos = (start_row, start_col)
            next_dir = 'right'
            pos_a, pos_b = cal_pos_ab(current_pos, next_dir)
            if digit_pos_legal(pos_a, nrow, ncol):
                remain[pos_a] = minus_one(remain[pos_a])
            if digit_pos_legal(pos_b, nrow, ncol):
                remain[pos_b] = minus_one(remain[pos_b])
            next_pos = cal_next_pos(current_pos, next_dir)
            point_path.append(next_pos)
            dir_path.append(next_dir)
            current_pos = next_pos
            backtrack_type = 1
            initial = True
            first_right = True  # The first step is to the right

        if first_right and (not initial) and len(point_path) == 1:
            print("No solution to this puzzle!")
            return False

        initial = False

        if backtrack_type == 1:  # Explore this node for the first time (turn left)
            last_dir = dir_path[-1]  # The direction of the previous step
            next_dir = turn_left(last_dir)

            while (not remain_allow(remain, current_pos, next_dir)) or (not forbid_allow(current_pos, next_dir, row_forbid, col_forbid)): 
                next_dir = turn_right(next_dir)
                if last_dir == opposite_dir(next_dir):
                    break

            last_flag = 2 if last_dir == opposite_dir(next_dir) else 1

            if last_flag == 1:  # Have not yet tried all three directions, can continue to try
                next_pos = cal_next_pos(current_pos, next_dir)
                
                # update remain
                pos_a, pos_b = cal_pos_ab(current_pos, next_dir)
                if digit_pos_legal(pos_a, nrow, ncol):
                    remain[pos_a] = minus_one(remain[pos_a])
                if digit_pos_legal(pos_b, nrow, ncol):
                    remain[pos_b] = minus_one(remain[pos_b])

                point_path.append(next_pos)
                dir_path.append(next_dir)
                current_pos = next_pos

                if current_pos == (start_row, start_col):  # If a loop is formed
                    if not(('1' in remain) or ('2' in remain) or ('3' in remain)):  # Meet the numerical limits
                        construct_jump_solution(problem, point_path)
                        return True
                    else:
                        cur_dir_last_try = dir_path[-1]
                        backtrack_type = 2
                        pos_a, pos_b = cal_pos_ab(current_pos=point_path[-2], next_dir=dir_path[-1])
                        if digit_pos_legal(pos_a, nrow, ncol):
                            remain[pos_a] = add_one(remain[pos_a])
                        if digit_pos_legal(pos_b, nrow, ncol):
                            remain[pos_b] = add_one(remain[pos_b])
                        # Delete the last step
                        point_path = point_path[:-1]
                        dir_path = dir_path[:-1]
                        current_pos = point_path[-1]
                        # continue # Back to the 'while true' loop
            
                if current_pos in point_path[:-1]:  # The path is crossed, but not a loop
                    cur_dir_last_try = dir_path[-1]
                    backtrack_type = 2
                    pos_a, pos_b = cal_pos_ab(current_pos=point_path[-2], next_dir=dir_path[-1])
                    if digit_pos_legal(pos_a, nrow, ncol):
                        remain[pos_a] = add_one(remain[pos_a])
                    if digit_pos_legal(pos_b, nrow, ncol):
                        remain[pos_b] = add_one(remain[pos_b])
                    # Delete the last step
                    point_path = point_path[:-1]
                    dir_path = dir_path[:-1]
                    current_pos = point_path[-1]
                    # continue # Back to the 'while true' loop

            else:  # last_flag == 2, have tried all three directions and failed
                cur_dir_last_try = dir_path[-1]
                backtrack_type = 2
                pos_a, pos_b = cal_pos_ab(current_pos=point_path[-2], next_dir=dir_path[-1])
                if digit_pos_legal(pos_a, nrow, ncol):
                    remain[pos_a] = add_one(remain[pos_a])
                if digit_pos_legal(pos_b, nrow, ncol):
                    remain[pos_b] = add_one(remain[pos_b])
                # Delete the last step
                point_path = point_path[:-1]
                dir_path = dir_path[:-1]
                current_pos = point_path[-1]
                # continue # Back to the 'while true' loop

        if backtrack_type == 2:  # Not the first time to explore this node (turn right according to the last direction that have tried)
            last_dir = dir_path[-1]
            next_dir = turn_right(cur_dir_last_try)

            if not (last_dir == opposite_dir(next_dir)):
                while (not remain_allow(remain, current_pos, next_dir)) or (not forbid_allow(current_pos, next_dir, row_forbid, col_forbid)):
                    next_dir = turn_right(next_dir)
                    if last_dir == opposite_dir(next_dir):
                        break
            
            last_flag = 2 if last_dir == opposite_dir(next_dir) else 1

            if last_flag == 1:
                next_pos = cal_next_pos(current_pos, next_dir)
            
                pos_a, pos_b = cal_pos_ab(current_pos, next_dir)
                if digit_pos_legal(pos_a, nrow, ncol):
                    remain[pos_a] = minus_one(remain[pos_a])
                if digit_pos_legal(pos_b, nrow, ncol):
                    remain[pos_b] = minus_one(remain[pos_b])

                point_path.append(next_pos)
                dir_path.append(next_dir)
                current_pos = next_pos
                backtrack_type = 1

                if current_pos == (start_row, start_col):  # If a loop is formed
                    if not(('1' in remain) or ('2' in remain) or ('3' in remain)):
                        construct_jump_solution(problem, point_path)
                        return True
                    else:
                        cur_dir_last_try = dir_path[-1]
                        backtrack_type = 2
                        pos_a, pos_b = cal_pos_ab(current_pos=point_path[-2], next_dir=dir_path[-1])
                        if digit_pos_legal(pos_a, nrow, ncol):
                            remain[pos_a] = add_one(remain[pos_a])
                        if digit_pos_legal(pos_b, nrow, ncol):
                            remain[pos_b] = add_one(remain[pos_b])
                        # Delete the last step
                        point_path = point_path[:-1]
                        dir_path = dir_path[:-1]
                        current_pos = point_path[-1]
                        # continue
            
                if current_pos in point_path[:-1]:  # The path is crossed, but not a loop
                    cur_dir_last_try = dir_path[-1]
                    backtrack_type = 2
                    pos_a, pos_b = cal_pos_ab(current_pos=point_path[-2], next_dir=dir_path[-1])
                    if digit_pos_legal(pos_a, nrow, ncol):
                        remain[pos_a] = add_one(remain[pos_a])
                    if digit_pos_legal(pos_b, nrow, ncol):
                        remain[pos_b] = add_one(remain[pos_b])
                    # Delete the last step
                    point_path = point_path[:-1]
                    dir_path = dir_path[:-1]
                    current_pos = point_path[-1]
                    # continue

            else:  # last_flag == 2, have tried all three directions and failed
                cur_dir_last_try = dir_path[-1]
                backtrack_type = 2
                pos_a, pos_b = cal_pos_ab(current_pos=point_path[-2], next_dir=dir_path[-1])
                if digit_pos_legal(pos_a, nrow, ncol):
                    remain[pos_a] = add_one(remain[pos_a])
                if digit_pos_legal(pos_b, nrow, ncol):
                    remain[pos_b] = add_one(remain[pos_b])
                # Delete the last step
                point_path = point_path[:-1]
                dir_path = dir_path[:-1]
                current_pos = point_path[-1]
                # continue


if __name__ == '__main__':
    # overall_limit = '1133'
    # overall_limit = '222333'  #2*3
    # overall_limit = '22*232321*32' # 3*4
    # overall_limit = '323*221**3233213' # 4*4
    # overall_limit = '22****3*223**23***3*3****'
    # overall_limit = '*******2**222*****3**33*3' # 5*5
    # nrow, ncol = 3, 4
    # problem = slitherlink(nrow, ncol, constraint=np.array(list(overall_limit)).reshape(nrow, ncol))

    problem = generate_problem_from_url(url='http://www.puzzle-loop.com/?v=0&size=0')
    problem.print_problem()

    start_time = time.time()
    if backjumping_solve(problem):
        print('success')
    end_time = time.time()
    print()
    print('start time: {}'.format(start_time))
    print('end time: {}'.format(end_time))
    print('time cost: {}'.format(end_time-start_time))
    # problem.print_solution()


