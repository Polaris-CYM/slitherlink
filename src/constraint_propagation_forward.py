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
    if remain[0, 0] == '1':
        row_forbid.append((0, 0))
        col_forbid.append((0, 0))
    if remain[0, ncol - 1] == '1':
        row_forbid.append((0, ncol - 1))
        col_forbid.append((0, ncol))
    if remain[nrow - 1, 0] == '1':
        row_forbid.append((nrow, 0))
        col_forbid.append((nrow - 1, 0))
    if remain[nrow - 1, ncol - 1] == '1':
        row_forbid.append((nrow, ncol - 1))
        col_forbid.append((nrow - 1, ncol))

    # situation 2: two '3' are connected
    for row in range(nrow):
        for col in range(ncol - 1):
            if remain[row, col] == '3' and remain[row, col + 1] == '3':
                if col_solution_pos_legal((row - 1, col + 1), nrow, ncol):
                    col_forbid.append((row - 1, col + 1))
                if col_solution_pos_legal((row + 1, col + 1), nrow, ncol):
                    col_forbid.append((row + 1, col + 1))
    for row in range(nrow - 1):
        for col in range(ncol):
            if remain[row, col] == '3' and remain[row + 1, col] == '3':
                if row_solution_pos_legal((row + 1, col - 1), nrow, ncol):
                    row_forbid.append((row + 1, col - 1))
                if row_solution_pos_legal((row + 1, col + 1), nrow, ncol):
                    row_forbid.append((row + 1, col + 1))

    return row_forbid, col_forbid
    

def find_force(remain):
    nrow, ncol = remain.shape
    row_force = []  # row_force = [(r1,c1), (r2,c2), ...] indicates that problem.row_solution(ri,ci) must be part of the solution
    col_force = []

    # situation 1: digit '3' is at the corner
    if remain[0,0] == '3':
        row_force.append((0,0))
        col_force.append((0,0))
    if remain[0,ncol-1] == '3':
        row_force.append((0,ncol-1))
        col_force.append((0,ncol))
    if remain[nrow-1,0] == '3':
        row_force.append((nrow,0))
        col_force.append((nrow-1,0))
    if remain[nrow-1,ncol-1] == '3':
        row_force.append((nrow,ncol-1))
        col_force.append((nrow-1,ncol))
    
    # situation 2: digit '2' is at the corner
    if remain[0,0] == '2':
        row_force.append((0,1))
        col_force.append((1,0))
    if remain[0,ncol-1] == '2':
        row_force.append((0,ncol-2))
        col_force.append((1,ncol))
    if remain[nrow-1,0] == '2':
        row_force.append((nrow,1))
        col_force.append((nrow-2,0))
    if remain[nrow-1,ncol-1] == '2':
        row_force.append((nrow,ncol-2))
        col_force.append((nrow-2,ncol))

    best_case = False
    # situation 3: '0' and '3' are connected
    for row in range(nrow):
        for col in range(ncol):
            if remain[row,col] == '3':
                if digit_pos_legal((row-1,col), nrow, ncol):
                    if remain[row-1,col] == '0':
                        best_case = True
                        col_force.append((row,col))
                        col_force.append((row,col+1))
                        row_force.append((row+1,col))
                        if row_solution_pos_legal((row,col-1), nrow, ncol):
                            row_force.append((row,col-1))
                        if row_solution_pos_legal((row,col+1), nrow, ncol):
                            row_force.append((row,col+1))
                if digit_pos_legal((row+1,col), nrow, ncol):
                    if remain[row+1,col] == '0':
                        best_case = True
                        col_force.append((row,col))
                        col_force.append((row,col+1))
                        row_force.append((row,col))
                        if row_solution_pos_legal((row+1,col-1), nrow, ncol):
                            row_force.append((row+1,col-1))
                        if row_solution_pos_legal((row+1,col+1), nrow, ncol):
                            row_force.append((row+1,col+1))
                if digit_pos_legal((row,col-1), nrow, ncol):
                    if remain[row,col-1] == '0':
                        best_case = True
                        row_force.append((row,col))
                        row_force.append((row+1,col))
                        col_force.append((row,col+1))
                        if col_solution_pos_legal((row-1,col), nrow, ncol):
                            col_force.append((row-1,col))
                        if col_solution_pos_legal((row+1,col), nrow, ncol):
                            col_force.append((row+1,col))
                if digit_pos_legal((row,col+1), nrow, ncol):
                    if remain[row,col+1] == '0':
                        best_case = True
                        row_force.append((row,col))
                        row_force.append((row+1,col))
                        col_force.append((row,col))
                        if col_solution_pos_legal((row-1,col+1), nrow, ncol):
                            col_force.append((row-1,col+1))
                        if col_solution_pos_legal((row+1,col+1), nrow, ncol):
                            col_force.append((row+1,col+1))

    # situation 4: a '3' is adjacent to a '0' diagonally
    for row in range(nrow):
        for col in range(ncol):
            if remain[row, col] == '3':
                if digit_pos_legal((row + 1, col + 1), nrow, ncol):
                    if remain[row + 1, col + 1] == '0':
                        col_force.append((row, col + 1))
                        row_force.append((row + 1, col))
                if digit_pos_legal((row + 1, col - 1), nrow, ncol):
                    if remain[row + 1, col - 1] == '0':
                        col_force.append((row, col))
                        row_force.append((row + 1, col))
                if digit_pos_legal((row - 1, col - 1), nrow, ncol):
                    if remain[row - 1, col - 1] == '0':
                        col_force.append((row, col))
                        row_force.append((row, col))
                if digit_pos_legal((row - 1, col + 1), nrow, ncol):
                    if remain[row - 1, col + 1] == '0':
                        col_force.append((row, col + 1))
                        row_force.append((row, col))

    # situation 5: a '3' is adjacent to a '3' diagonally
    for row in range(nrow):
        for col in range(ncol):
            if remain[row, col] == '3':
                if digit_pos_legal((row + 1, col + 1), nrow, ncol):
                    if remain[row + 1, col + 1] == '3':
                        col_force.append((row, col))
                        row_force.append((row, col))
                        col_force.append((row + 1, col + 2))
                        row_force.append((row + 2, col + 1))
                if digit_pos_legal((row + 1, col - 1), nrow, ncol):
                    if remain[row + 1, col - 1] == '3':
                        col_force.append((row, col + 1))
                        row_force.append((row, col))
                        col_force.append((row + 1, col - 1))
                        row_force.append((row + 2, col - 1))
                if digit_pos_legal((row - 1, col - 1), nrow, ncol):
                    if remain[row - 1, col - 1] == '3':
                        col_force.append((row, col + 1))
                        row_force.append((row + 1, col))
                        col_force.append((row - 1, col - 1))
                        row_force.append((row - 1, col - 1))
                if digit_pos_legal((row - 1, col + 1), nrow, ncol):
                    if remain[row - 1, col + 1] == '3':
                        col_force.append((row, col))
                        row_force.append((row + 1, col))
                        col_force.append((row - 1, col + 2))
                        row_force.append((row - 1, col + 1))

    # situation 6: two 3s are in the same diagonal, but separated by a '2'
    for row in range(nrow):
        for col in range(ncol):
            if remain[row, col] == '3':
                if digit_pos_legal((row + 1, col + 1), nrow, ncol) and digit_pos_legal((row + 2, col + 2), nrow, ncol):
                    if remain[row + 1, col + 1] == '2' and remain[row + 2, col + 2] == '3':
                        col_force.append((row, col))
                        row_force.append((row, col))
                        col_force.append((row + 2, col + 3))
                        row_force.append((row + 3, col + 2))
                if digit_pos_legal((row + 1, col - 1), nrow, ncol) and digit_pos_legal((row + 2, col - 2), nrow, ncol):
                    if remain[row + 1, col - 1] == '2' and remain[row + 2, col - 2] == '3':
                        col_force.append((row, col + 1))
                        row_force.append((row, col))
                        col_force.append((row + 2, col - 2))
                        row_force.append((row + 3, col - 2))
                if digit_pos_legal((row - 1, col - 1), nrow, ncol) and digit_pos_legal((row - 2, col - 2), nrow, ncol):
                    if remain[row - 1, col - 1] == '2' and remain[row - 2, col - 2] == '3':
                        col_force.append((row, col + 1))
                        row_force.append((row + 1, col))
                        col_force.append((row - 2, col - 2))
                        row_force.append((row - 2, col - 2))
                if digit_pos_legal((row - 1, col + 1), nrow, ncol) and digit_pos_legal((row - 2, col + 2), nrow, ncol):
                    if remain[row - 1, col + 1] == '2' and remain[row - 2, col + 2] == '3':
                        col_force.append((row, col))
                        row_force.append((row + 1, col))
                        col_force.append((row - 2, col + 3))
                        row_force.append((row - 2, col + 2))
    
    return row_force, col_force


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
    

def get_force_direction_list(remain):
    """
    Get the nodes that must be included in the solution and the direction they must go
    """
    nrow, ncol = remain.shape
    row_force, col_force = find_force(remain)
    # nrow+1: row coordinate of the node
    # ncol+1: column coordinate of the node
    # 4: 4 directions: 0-up,1-down,2-left,3-right
    force_direction_list = np.zeros(shape=(nrow+1,ncol+1,4))

    for row_pos in row_force:
        force_direction_list[row_pos[0], row_pos[1], 3] = 1  # The node from (row_pos[0], row_pos[1]) must go to the right
        force_direction_list[row_pos[0], row_pos[1]+1, 2] = 1
    for col_pos in col_force:
        force_direction_list[col_pos[0], col_pos[1], 1] = 1
        force_direction_list[col_pos[0]+1, col_pos[1], 0] = 1
    return force_direction_list


def get_force_direction(current_pos, force_direction_list):
    force_dir = []
    if force_direction_list[current_pos[0], current_pos[1], 0] == 1:
        force_dir.append('up')
    if force_direction_list[current_pos[0], current_pos[1], 1] == 1:
        force_dir.append('down')
    if force_direction_list[current_pos[0], current_pos[1], 2] == 1:
        force_dir.append('left')
    if force_direction_list[current_pos[0], current_pos[1], 3] == 1:
        force_dir.append('right')
    return force_dir

def construct_prop_solution(problem, point_path):
    """
    Construct the solution of the puzzle based on point_path
    """
    # Initialize to 0
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
    print('FORWARD CHECKING SOLUTION:')
    problem.print_solution()
    return True

def constraint_propagation_forward(problem):
    """
    Use the constraint_propagation combined with forward checking algorithm to solve the puzzle
    """
    nrow, ncol = problem.nrow, problem.ncol
    start_row, start_col = find_start(problem)  # Find the start point (function find_start is in backtracking.py)
    point_path = [(start_row, start_col)] # point_path array: store the point path of the final solution
    dir_path = []  # Direction from the start point
    remain = problem.constraint.copy()  # Maintain a copy of the constraint, which can be increased or decreased at any time
    row_forbid, col_forbid = find_forbid(remain)
    force_direction_list = get_force_direction_list(remain)
    current_pos = (start_row, start_col)  # Position of current node

    first_right = False # whether the first step is to the right

    temp_force_dir = get_force_direction(current_pos, force_direction_list)
    if len(temp_force_dir) > 0:
        next_dir = temp_force_dir[0]
        if next_dir == 'right':
            first_right = True
    else:
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
            temp_force_dir = get_force_direction(current_pos, force_direction_list)
            # can't go back
            if opposite_dir(last_dir) in temp_force_dir:
                temp_force_dir.remove(opposite_dir(last_dir))
            
            if len(temp_force_dir) > 0:
                next_dir = temp_force_dir[0]
            else:
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
                        construct_prop_solution(problem, point_path)
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

            if last_flag == 1 and (last_dir not in get_force_direction(current_pos, force_direction_list)):
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
                        construct_prop_solution(problem, point_path)
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
    # overall_limit = '*1****1*12*1**3**0*13*33*'
    # overall_limit = '*3*33***21**3**2**2**33*3'
    # nrow, ncol = 5, 5
    # problem = slitherlink(nrow, ncol, constraint=np.array(list(overall_limit)).reshape(nrow, ncol))

    problem = generate_problem_from_url(url='http://www.puzzle-loop.com/?v=0&size=0')
    problem.print_problem()

    start_time = time.time()
    if constraint_propagation_forward(problem):
        print('success')
    end_time = time.time()
    print()
    print('start time: {}'.format(start_time))
    print('end time: {}'.format(end_time))
    print('time cost: {}'.format(end_time-start_time))
    # problem.print_solution()
