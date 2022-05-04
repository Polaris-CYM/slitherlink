import time
import numpy as np
from problem_define import slitherlink
from generate_problem import *


def find_start(problem):
    """
    Find where to start, return the index of row and col
    """
    already_find = False
    for i in range(problem.nrow):
        for j in range(problem.ncol):
            # Find the first "3" in the order from top to bottom, left to right, use its upper left corner as the
            # starting point. This point must be in the final path because the constraint is "3"
            if problem.constraint[i, j] == '3':
                start_row = i
                start_col = j
                already_find = True
                break
        if already_find:
            break
    if already_find:
        return start_row, start_col
    else:
        raise ValueError("There is no '3' in the number constraint of the puzzle")


def judge_direction(pos1, pos2):
    """
    Given two positions, return the direction from pos1 to pos2
    :param pos1: a tuple, (pos1_row, pos1_col)
    :param pos2: a tuple, (pos2_row, pos2_col)
    """
    if not(isinstance(pos1, tuple) and isinstance(pos2, tuple)):
        raise TypeError('In function judge_direction: pos1 or pos2 is not a tuple')

    r1, c1 = pos1
    r2, c2 = pos2
    if r1 == r2 and c1 == c2 + 1:
        return 'left'
    elif r1 == r2 and c1 == c2 - 1:
        return 'right'
    elif r1 == r2 + 1 and c1 == c2:
        return 'up'
    elif r1 == r2 - 1 and c1 == c2:
        return 'down'
    else:
        return ValueError('In function judge_direction: value of pos1 or pos2 is not valid')


def turn_left(dir):
    """
    Based on the direction of the previous step, determine which direction to go if the next step is to turn left
    :param dir: left/right/up/down, the direction of the previous step
    """
    if dir == 'left':
        return 'down'
    elif dir == 'right':
        return 'up'
    elif dir == 'down':
        return 'right'
    elif dir == 'up':
        return 'left'
    else:
        raise ValueError('In function turn_left: the value of dir', dir, 'is not valid')


def turn_right(dir):
    """
    Based on the direction of the previous step, determine which direction to go if the next step is to turn right
    """
    if dir == 'left':
        return 'up'
    elif dir == 'up':
        return 'right'
    elif dir == 'right':
        return 'down'
    elif dir == 'down':
        return 'left'
    else:
        raise ValueError('In function turn_right: the value of dir', dir, 'is not valid')


def opposite_dir(dir):
    if dir == 'down':
        return 'up'
    if dir == 'up':
        return 'down'
    if dir == 'left':
        return 'right'
    if dir == 'right':
        return 'left'


def remain_allow(remain, current_pos, next_dir):
    """
    Determine whether it is legal to go from current_pos to next_dir, i.e., whether it satisfies:
    1. the newly reached point does not exceed the puzzle boundary
    2. no remain value on both sides of the path is 0 in this direction
    """
    nrow, ncol = remain.shape
    next_pos = cal_next_pos(current_pos, next_dir)
    if point_pos_legal(next_pos, nrow, ncol) == False:
        return False

    pos_a, pos_b = cal_pos_ab(current_pos, next_dir)
    if digit_pos_legal(pos_a, nrow, ncol):
        if remain[pos_a] == '0':
            return False
    if digit_pos_legal(pos_b, nrow, ncol):
        if remain[pos_b] == '0':
            return False

    return True


def cal_next_pos(pos, dir):
    """
    Calculate the next position based on dir
    """
    if not isinstance(pos, tuple):
        raise TypeError('In function cal_next_pos: pos is not a tuple')
    if dir not in ['left', 'right', 'up', 'down']:
        raise TypeError('In function cal_next_pos: dir is not legal')

    r, c = pos
    if dir == 'left':
        return (r, c-1)
    if dir == 'right':
        return (r, c+1)
    if dir == 'up':
        return (r-1, c)
    else:  # dir == 'down'
        return (r+1, c)


def cal_pos_ab(current_pos, next_dir):
    """
    Calculate the position of the numbers on either side of the upcoming step
    """
    # next_pos/current_pos: position of the node, pos_a/pos_b: position of the digit
    next_pos = cal_next_pos(current_pos, next_dir)

    if next_dir == 'down':
        pos_a = (next_pos[0] - 1, next_pos[1] - 1)
        pos_b = (next_pos[0] - 1, next_pos[1])
    elif next_dir == 'up':
        pos_a = (next_pos[0], next_pos[1] - 1)
        pos_b = (next_pos[0], next_pos[1])
    elif next_dir == 'left':
        pos_a = (next_pos[0] - 1, next_pos[1])
        pos_b = (next_pos[0], next_pos[1])
    elif next_dir == 'right':
        pos_a = (next_pos[0] - 1, next_pos[1] - 1)
        pos_b = (next_pos[0], next_pos[1] - 1)

    return pos_a, pos_b


def point_pos_legal(pos, nrow, ncol):
    """
    Determine whether the current node is out of bounds
    """
    r, c = pos
    if 0 <= r <= nrow and 0 <= c <= ncol:
        return True
    else:
        return False


def digit_pos_legal(pos, nrow, ncol):
    """
    Determine whether the position of the currently located digit is out of bounds
    """
    r, c = pos
    if 0 <= r < nrow and 0 <= c < ncol:
        return True
    else:
        return False


def minus_one(num):
    if num == '1':
        return '0'
    elif num == '2':
        return '1'
    elif num == '3':
        return '2'
    elif num == '*':
        return '*'


def add_one(num):
    if num == '0':
        return '1'
    elif num == '1':
        return '2'
    elif num == '2':
        return '3'
    elif num == '*':
        return '*'


def construct_track_solution(problem, point_path):
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
    print('BACKTRACKING SOLUTION:')
    problem.print_solution()
    return True


def backtracking_solve(problem):
    """
    use the backtracking search algorithm to solve the problem
    """
    nrow, ncol = problem.nrow, problem.ncol
    start_row, start_col = find_start(problem)  # Find the start point
    point_path = [(start_row, start_col)]  # point_path array: store the point path of the final solution
    dir_path = []  # Direction from the start point
    remain = problem.constraint.copy()  # Maintain a copy of the constraint, which can be increased or decreased at any time
    current_pos = (start_row, start_col)  # Position of current node

    first_right = False  # whether the first step is to the right

    # Go down first
    next_dir = 'down'
    # If it can't go down (the only possibility is that 3 is connected with 0), go to the right
    if not remain_allow(remain, current_pos, next_dir):
        next_dir = 'right'
        first_right = True

    # Update 'remain' matrix
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
        if initial == False and len(point_path) == 1 and first_right == False:
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

            while not remain_allow(remain, current_pos, next_dir):
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
                        construct_track_solution(problem, point_path)
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
                while not remain_allow(remain, current_pos, next_dir):
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
                        construct_track_solution(problem, point_path)
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
    # overall_limit = '222333'  # 2*3
    # overall_limit = '22*232321*32' # 3*4
    # overall_limit = '323*221**3233213' # 4*4
    # overall_limit = '22****3*223**23***3*3****'
    # overall_limit = '*******2**222*****3**33*3' # 5*5
    # overall_limit = '2122223211222223232333212022222323221210222232233'
    # overall_limit = '******33**1***12****20***1**11*2****' # 6*6
    # nrow, ncol = 6,6
    # problem = slitherlink(nrow, ncol, constraint=np.array(list(overall_limit)).reshape(nrow, ncol))

    problem = generate_problem_from_url(url='http://www.puzzle-loop.com/?v=0&size=0')
    problem.print_problem()

    start_time = time.time()
    if backtracking_solve(problem):
        print('success')
    end_time = time.time()
    print()
    print('start time: {}'.format(start_time))
    print('end time: {}'.format(end_time))
    print('time cost: {}'.format(end_time-start_time))
