import time
import numpy as np
from problem_define import slitherlink
from generate_problem import *

def find_start(problem):
    '''
    find where to start, return the index of row and col
    '''
    already_find = False
    for i in range(problem.nrow):
        for j in range(problem.ncol):
            if problem.constraint[i, j] == '3':  # Find the first "3" in the order from top to bottom, left to right,
                                                 # because 3 has more constraints and fewer possibilities.
                start_row = i
                start_col = j
                already_find = True
                break
        if already_find:
            break
    if already_find:
        return start_row, start_col
    else:
        raise ValueError("problem约束中没有3，需要进一步修改回溯算法中的find_start函数")

def judge_direction(pos1, pos2):
    '''
    given two positions, return what direction is from pos1 to pos2
    :param pos1: a tuple, (pos1_x, pos1_y)
    :param pos2: a tuple, (pos2_x, pos2_y)
    '''
    if not(isinstance(pos1, tuple) and isinstance(pos2, tuple)):
        raise TypeError('In function judge_direction: pos1 or pos2 is not tuple')

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
    '''
    determine what direction should be selcted based on the path before
    according to the strtegy that always turn left
    :param dir: left/right/up/down, the direction of the last part of the path
    '''
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
    '''
    return the direction on the right side
    '''
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
    '''
    在当前的current pos，往next_dir方向走，是否合法，即是否满足
    1. 不超出棋盘范围
    2. 往这个方向走，路径两边的remain值不为0
    '''
    nrow, ncol = remain.shape
    next_pos = cal_next_pos(current_pos, next_dir)
    if point_pos_legal(next_pos, nrow, ncol) == False:
        # 如果下一个落脚点超出了棋盘范围
        return False # 不能朝着这个方向走

    pos_a, pos_b = cal_pos_ab(current_pos, next_dir)
    if digit_pos_legal(pos_a, nrow, ncol):
        if remain[pos_a] == '0':
            return False
    if digit_pos_legal(pos_b, nrow, ncol):
        if remain[pos_b] == '0':
            return False

    return True

def last_possible_direction(dir):
    '''
    return the direction on the left side
    '''
    if dir == 'left':
        return 'down'
    elif dir == 'up':
        return 'left'
    elif dir == 'right':
        return 'up'
    elif dir == 'down':
        return 'right'
    else:
        raise ValueError('In function last_possible_direction: the value of dir', dir, 'is not valid')

def cal_next_pos(pos, dir):
    '''
    calculate the next position based on dir
    '''
    if not isinstance(pos, tuple):
        raise TypeError('In function cal_next_pos: pos is not tuple')
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

def point_pos_legal(pos, nrow, ncol):
    '''
    return True or False
    '''
    r, c = pos
    if 0 <= r <= nrow and 0 <= c <= ncol:
        return True
    else:
        return False

def digit_pos_legal(pos, nrow, ncol):
    '''
    return True or False
    '''
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

def output_status(backtrack_type, remain, point_path, dir_path, current_pos, next_dir, here):
    '''print('go here', here)
    print(remain)
    print('backtrack_type:', backtrack_type)
    print('point path:', point_path)
    print('dir path:', dir_path)
    print('current_pos:', current_pos)
    print('next_dir:', next_dir)'''
    print()

def cal_pos_ab(current_pos, next_dir):
    '''
    计算下一步两边的数字的index
    '''
    next_pos = cal_next_pos(current_pos, next_dir)

    if next_dir == 'down':
        pos_a = (next_pos[0]-1, next_pos[1]-1)
        pos_b = (next_pos[0]-1, next_pos[1])
    elif next_dir == 'up':
        pos_a = (next_pos[0], next_pos[1]-1)
        pos_b = (next_pos[0], next_pos[1])
    elif next_dir == 'left':
        pos_a = (next_pos[0]-1, next_pos[1])
        pos_b = (next_pos[0], next_pos[1])
    elif next_dir == 'right':
        pos_a = (next_pos[0]-1, next_pos[1]-1)
        pos_b = (next_pos[0], next_pos[1]-1)
    
    return pos_a, pos_b

def construct_track_solution(problem, point_path):
    '''
    根据point_path构建problem的解
    '''
    # 初始化为0
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
    '''
    use the backtracking search algorithm to solve the problem
    '''
    nrow, ncol = problem.nrow, problem.ncol
    start_row, start_col = find_start(problem)  # 从这个点开始画圈，因为旁边就有3，所以画的第一条边大概率是解的一部分
    point_path = [(start_row, start_col)]  # 维护loop solution的点集合，点是square的顶点
    dir_path = []  # 从start这个点开始的方向
    remain = problem.constraint.copy()  # 维护一个constraint的副本，随时增减
    current_pos = (start_row, start_col)  # 当前位置
    
    # 先向下走，向下画loop
    next_dir = 'down'
    if not remain_allow(remain, current_pos, next_dir): # 如果不能向下走（只可能因为03连一起）就向右边走
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
    # type=1是初始时刻往下走，或者新探索某个点
    # type=2是以前走过，后来这条路走不通，要换个方向
    initial = True
    first_right = False

    while True:
        if initial == False and len(point_path) == 1: 
            # 如果回退到最初的点
            # 那么说明第一步不该走down
            # 以下修改相关信息，重新循环
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
            first_right = True # 第一步是向右边走的

        if first_right and (not initial) and len(point_path) == 1:
            # 如果第一步已经向右走，且又回溯回起始点，那么说明此题无解
            print("There is no solution")
            return False

        initial = False

        if backtrack_type == 1: # 初次探索这个点
            # output_status(backtrack_type, remain, point_path, dir_path, current_pos, next_dir, here=1)

            last_dir = dir_path[-1] # 上一步的方向
            next_dir = turn_left(last_dir) # 根据上一步方向向左转

            while not remain_allow(remain, current_pos, next_dir): # 只有当 remain/界面大小 不允许的情况才会循环
                next_dir = turn_right(next_dir) # 向右转，尝试下一个方向
                if last_dir == opposite_dir(next_dir):
                    break

            last_flag = 2 if last_dir == opposite_dir(next_dir) else 1

            if last_flag == 1: # 旋转旋转，还没转一圈，还可以继续尝试
                # output_status(backtrack_type, remain, point_path, dir_path, current_pos, next_dir, here=2)
                next_pos = cal_next_pos(current_pos, next_dir) # 根据对应方向，计算下一个落脚点
                
                # update remain
                pos_a, pos_b = cal_pos_ab(current_pos, next_dir)
                if digit_pos_legal(pos_a, nrow, ncol):
                    remain[pos_a] = minus_one(remain[pos_a])
                if digit_pos_legal(pos_b, nrow, ncol):
                    remain[pos_b] = minus_one(remain[pos_b])

                point_path.append(next_pos)
                dir_path.append(next_dir)
                current_pos = next_pos

                if current_pos == (start_row, start_col): # 构成一个loop了
                    # output_status(backtrack_type, remain, point_path, dir_path, current_pos, next_dir, here=3)
                    if not(('1' in remain) or ('2' in remain) or ('3' in remain)): # 合法解
                        # output_status(backtrack_type, remain, point_path, dir_path, current_pos, next_dir, here=4)
                        construct_track_solution(problem, point_path)
                        return True
                    else: # 构成一个循环，但是还有数字没有消灭
                        # output_status(backtrack_type, remain, point_path, dir_path, current_pos, next_dir, here=5)
                        cur_dir_last_try = dir_path[-1] # current_pos出发的 刚刚尝试过的方向，对于回溯后下一次来说 是last try
                        backtrack_type = 2
                        # 更新remain，加回1
                        pos_a, pos_b = cal_pos_ab(current_pos=point_path[-2], next_dir=dir_path[-1])
                        if digit_pos_legal(pos_a, nrow, ncol):
                            remain[pos_a] = add_one(remain[pos_a])
                        if digit_pos_legal(pos_b, nrow, ncol):
                            remain[pos_b] = add_one(remain[pos_b])
                        # 更新path
                        point_path = point_path[:-1] # 删掉最后一步
                        dir_path = dir_path[:-1] # 删掉最后一步
                        current_pos = point_path[-1]
                        # continue # 回到while true循环
            
                if current_pos in point_path[:-1]: # 交叉了，但不是一个loop（因为如果是loop，就会进入前面一个if分支）
                    # output_status(backtrack_type, remain, point_path, dir_path, current_pos, next_dir, here=6)
                    cur_dir_last_try = dir_path[-1]
                    backtrack_type = 2
                    # 更新remain，加回1
                    pos_a, pos_b = cal_pos_ab(current_pos=point_path[-2], next_dir=dir_path[-1])
                    if digit_pos_legal(pos_a, nrow, ncol):
                        remain[pos_a] = add_one(remain[pos_a])
                    if digit_pos_legal(pos_b, nrow, ncol):
                        remain[pos_b] = add_one(remain[pos_b])
                    # 更新path
                    point_path = point_path[:-1] # 删掉最后一步
                    dir_path = dir_path[:-1] # 删掉最后一步
                    current_pos = point_path[-1]
                    # continue # 回到while true循环

            else: # last_flag == 2, 转了一圈转回来了，要回溯了，此时next_dir打算往回走了
                # output_status(backtrack_type, remain, point_path, dir_path, current_pos, next_dir, here=7)
                cur_dir_last_try = dir_path[-1]
                backtrack_type = 2
                # 更新remain
                pos_a, pos_b = cal_pos_ab(current_pos=point_path[-2], next_dir=dir_path[-1])
                if digit_pos_legal(pos_a, nrow, ncol):
                    remain[pos_a] = add_one(remain[pos_a])
                if digit_pos_legal(pos_b, nrow, ncol):
                    remain[pos_b] = add_one(remain[pos_b])
                # 更新path
                point_path = point_path[:-1] # 删掉最后一步
                dir_path = dir_path[:-1] # 删掉最后一步
                current_pos = point_path[-1]
                # continue # 回到while true循环

        if backtrack_type == 2: # 不是第一次探索这个点，根据上一次选的方向继续调整
            # output_status(backtrack_type, remain, point_path, dir_path, current_pos, next_dir, here=8)
            last_dir = dir_path[-1] # 上一步的方向
            next_dir = turn_right(cur_dir_last_try) # 根据上一步方向向左转

            if not (last_dir == opposite_dir(next_dir)):
                while not remain_allow(remain, current_pos, next_dir): # 只有当 remain/界面大小 不允许的情况才会循环
                    next_dir = turn_right(next_dir)
                    if last_dir == opposite_dir(next_dir):
                        break
            
            last_flag = 2 if last_dir == opposite_dir(next_dir) else 1

            if last_flag == 1:
                # output_status(backtrack_type, remain, point_path, dir_path, current_pos, next_dir, here=9)
                next_pos = cal_next_pos(current_pos, next_dir) # 根据对应方向，计算下一个落脚点
            
                # update remain
                pos_a, pos_b = cal_pos_ab(current_pos, next_dir)
                if digit_pos_legal(pos_a, nrow, ncol):
                    remain[pos_a] = minus_one(remain[pos_a])
                if digit_pos_legal(pos_b, nrow, ncol):
                    remain[pos_b] = minus_one(remain[pos_b])

                point_path.append(next_pos)
                dir_path.append(next_dir)
                current_pos = next_pos
                # 还原type
                backtrack_type = 1

                if current_pos == (start_row, start_col): # 构成一个loop了
                    # output_status(backtrack_type, remain, point_path, dir_path, current_pos, next_dir, here=10)
                    if not(('1' in remain) or ('2' in remain) or ('3' in remain)): # 合法解
                        # output_status(backtrack_type, remain, point_path, dir_path, current_pos, next_dir, here=11)
                        construct_track_solution(problem, point_path)
                        return True
                    else: # 构成一个循环，但是还有数字没有消灭
                        # output_status(backtrack_type, remain, point_path, dir_path, current_pos, next_dir, here=12)
                        cur_dir_last_try = dir_path[-1]
                        backtrack_type = 2
                        # 更新remain，加回1
                        pos_a, pos_b = cal_pos_ab(current_pos=point_path[-2], next_dir=dir_path[-1])
                        if digit_pos_legal(pos_a, nrow, ncol):
                            remain[pos_a] = add_one(remain[pos_a])
                        if digit_pos_legal(pos_b, nrow, ncol):
                            remain[pos_b] = add_one(remain[pos_b])
                        # 更新path
                        point_path = point_path[:-1] # 删掉最后一步
                        dir_path = dir_path[:-1] # 删掉最后一步
                        current_pos = point_path[-1]
                        # continue # 回到while true循环
            
                if current_pos in point_path[:-1]: # 交叉了，但不是一个loop（因为如果是loop，就会进入前面一个if分支）
                    # output_status(backtrack_type, remain, point_path, dir_path, current_pos, next_dir, here=13)
                    cur_dir_last_try = dir_path[-1]
                    backtrack_type = 2
                    # 更新remain，加回1
                    pos_a, pos_b = cal_pos_ab(current_pos=point_path[-2], next_dir=dir_path[-1])
                    if digit_pos_legal(pos_a, nrow, ncol):
                        remain[pos_a] = add_one(remain[pos_a])
                    if digit_pos_legal(pos_b, nrow, ncol):
                        remain[pos_b] = add_one(remain[pos_b])
                    # 更新path
                    point_path = point_path[:-1] # 删掉最后一步
                    dir_path = dir_path[:-1] # 删掉最后一步
                    current_pos = point_path[-1]
                    # continue # 回到while true循环

            else: # last_flag == 2, 转一圈转回来了
                # output_status(backtrack_type, remain, point_path, dir_path, current_pos, next_dir, here=14)
                cur_dir_last_try = dir_path[-1]
                backtrack_type = 2
                # 更新remain
                pos_a, pos_b = cal_pos_ab(current_pos=point_path[-2], next_dir=dir_path[-1])
                if digit_pos_legal(pos_a, nrow, ncol):
                    remain[pos_a] = add_one(remain[pos_a])
                if digit_pos_legal(pos_b, nrow, ncol):
                    remain[pos_b] = add_one(remain[pos_b])
                # 更新path
                point_path = point_path[:-1] # 删掉最后一步
                dir_path = dir_path[:-1] # 删掉最后一步
                current_pos = point_path[-1]
                # continue # 回到while true循环   

if __name__ == '__main__':
    # overall_limit = '1133'
    # overall_limit = '222333'  #2*3
    # overall_limit = '22*232321*32' # 3*4
    # overall_limit = '323*221**3233213' # 4*4
    # overall_limit = '22****3*223**23***3*3****'
    # overall_limit = '*******2**222*****3**33*3' # 5*5  这个例子要运行比较久
    # overall_limit = '3**23*****2*203*23***23*3'
    # overall_limit = '******33**1***12****20***1**11*2****' # 6*6
    # nrow, ncol = 5,5
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
    # problem.print_solution()
