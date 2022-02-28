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
            if problem.constraint[i,j] == '3':  # 找一个最早开始的3，因为3的约束条件多，所以尽量找一个大一点的数
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


def determine_what_direction(dir):
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
        raise ValueError('In function determine_what_direction: the value of dir', dir, 'is not valid')
    

def next_direction(dir):
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
        raise ValueError('In function next_direction: the value of dir', dir, 'is not valid')


def remain_allow(remain, current_pos, next_dir):
    '''
    在当前的current pos，往next_dir方向走，是否合法，即是否满足
    1. 不超出棋盘范围
    2. 往这个方向走，路径两边的remain值不为0
    '''
    nrow, ncol = remain.shape
    next_pos = cal_next_pos(current_pos, next_dir)
    if pos_legal(next_pos, nrow, ncol) == False:
        # 如果下一个落脚点超出了棋盘范围
        return False # 不能朝着这个方向走
    
    if next_dir == 'down':
        pos_a = (next_pos[0]-1, next_pos[1]-1)
        pos_b = (next_pos[0]-1, next_pos[1])
        if pos_legal2(pos_a, nrow, ncol):
            if remain[pos_a] in ['0']:
                return False
        if pos_legal2(pos_b, nrow, ncol):
            if remain[pos_b] in ['0']:
                return False
    if next_dir == 'up':
        pos_a = (next_pos[0], next_pos[1]-1)
        pos_b = (next_pos[0], next_pos[1])
        if pos_legal2(pos_a, nrow, ncol):
            if remain[pos_a] in ['0']:
                return False
        if pos_legal2(pos_b, nrow, ncol):
            if remain[pos_b] in ['0']:
                return False
    if next_dir == 'left':
        pos_a = (next_pos[0]-1, next_pos[1])
        pos_b = (next_pos[0], next_pos[1])
        if pos_legal2(pos_a, nrow, ncol):
            if remain[pos_a] in ['0']:
                return False
        if pos_legal2(pos_b, nrow, ncol):
            if remain[pos_b] in ['0']:
                return False
    if next_dir == 'right':
        pos_a = (next_pos[0]-1, next_pos[1]-1)
        pos_b = (next_pos[0], next_pos[1]-1)
        if pos_legal2(pos_a, nrow, ncol):
            if remain[pos_a] in ['0']:
                return False
        if pos_legal2(pos_b, nrow, ncol):
            if remain[pos_b] in ['0']:
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


def pos_legal(pos, nrow, ncol):
    '''
    return True or False
    '''
    r, c = pos
    if 0 <= r <= nrow and 0 <= c <= ncol:
        return True
    else:
        return False


def pos_legal2(pos, nrow, ncol):
    '''
    return True or False
    '''
    r, c = pos
    if 0 <= r < nrow and 0 <= c < ncol:
        return True
    else:
        return False


def minus_one(c):
    if c == '1':
        return '0'
    elif c == '2':
        return '1'
    elif c == '3':
        return '2'
    elif c == '*':
        return '*'


def opposite_dir(dir):
    if dir == 'down':
        return 'up'
    if dir == 'up':
        return 'down'
    if dir == 'left':
        return 'right'
    if dir == 'right':
        return 'left'


def output_status(remain, point_path, dir_path):
    print('\n*********')
    print(remain)
    print('point path:', point_path)
    print('dir path:', dir_path)
    print('***********\n')


def construct_solution(problem, point_path):
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
    pos = cal_next_pos(current_pos, 'down')
    point_path.append(pos)
    dir_path.append('down')

    # 更新remain
    remain[current_pos] = minus_one(remain[current_pos])
    start_flag = 0
    if pos_legal2((current_pos[0], current_pos[1]-1), nrow, ncol):
        if remain[current_pos[0], current_pos[1]-1] == '0':
            start_flag = 12345
        else:
            start_flag = 0
            remain[current_pos[0], current_pos[1]-1] = minus_one(remain[current_pos[0], current_pos[1]-1])
        
    current_pos = pos

    backtrack_type = 1  
    # type=1是初始时刻往下走，或者新探索某个点
    # type=2是以前走过，后来这条路走不通，要换个方向

    while True:
        output_status(remain, point_path, dir_path)

        if start_flag == 12345: # 第一步就不该向下走
            break
        
        if backtrack_type == 1: # 初次探索这个点
            last_dir = dir_path[-1] # 上一步的方向
            next_dir = determine_what_direction(last_dir) # 根据上一步方向向左转
            last_flag = 1

            while not remain_allow(remain, current_pos, next_dir): # 只有当 remain/界面大小 不允许的情况才会循环
                last_flag = 1
                next_dir = next_direction(next_dir)
                if last_dir == opposite_dir(next_dir):
                    last_flag = 2
                    break

            if last_flag == 1: # 旋转旋转，还没转一圈，还可以继续尝试
                next_pos = cal_next_pos(current_pos, next_dir) # 根据对应方向，计算下一个落脚点
            
                # update remain
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

                remain[pos_a] = minus_one(remain[pos_a])
                remain[pos_b] = minus_one(remain[pos_b])

                point_path.append(next_pos)
                dir_path.append(next_dir)
                current_pos = next_pos

                if next_pos == (start_row, start_col): # 构成一个loop了
                    if not(('1' in remain) or ('2' in remain) or ('3' in remain)): # 合法解
                        return construct_solution(problem, point_path)
                    else: # 构成一个循环，但是还有数字没有消灭
                        cur_dir_last_try = next_dir
                        backtrack_type = 2
                        point_path = point_path[:-1] # 删掉最后一步
                        dir_path = dir_path[:-1] # 删掉最后一步
                        continue # 回到while true循环
            
                if next_pos in point_path: # 交叉了，但不是一个loop（因为如果是loop，就会进入前面一个if分支）
                    cur_dir_last_try = next_dir
                    backtrack_type = 2
                    point_path = point_path[:-1] # 删掉最后一步
                    dir_path = dir_path[:-1] # 删掉最后一步
                    continue # 回到while true循环

            else: # last_flag == 2, 转了一圈转回来了，要回溯了
                cur_dir_last_try = dir_path[-1]
                backtrack_type = 2
                point_path = point_path[:-1] # 删掉最后一步
                dir_path = dir_path[:-1] # 删掉最后一步
                continue # 回到while true循环

        if backtrack_type == 2: # 不是第一次探索这个点，根据上一次选的方向继续调整
            last_dir = dir_path[-1] # 上一步的方向
            next_dir = next_direction(cur_dir_last_try) # 根据上一步方向向左转
            last_flag = 2 if last_dir == opposite_dir(next_dir) else 1

            while not remain_allow(remain, current_pos, next_dir): # 只有当 remain/界面大小 不允许的情况才会循环
                last_flag = 1
                next_dir = next_direction(next_dir)
                if last_dir == opposite_dir(next_dir):
                    last_flag = 2
                    break

            if last_flag == 1:
                next_pos = cal_next_pos(current_pos, next_dir) # 根据对应方向，计算下一个落脚点
            
                # update remain
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

                remain[pos_a] = minus_one(remain[pos_a])
                remain[pos_b] = minus_one(remain[pos_b])

                point_path.append(next_pos)
                dir_path.append(next_dir)
                current_pos = next_pos
                # 还原type
                backtrack_type = 1

                if next_pos == (start_row, start_col): # 构成一个loop了
                    if not(('1' in remain) or ('2' in remain) or ('3' in remain)): # 合法解
                        return construct_solution(problem, point_path)
                    else: # 构成一个循环，但是还有数字没有消灭
                        cur_dir_last_try = next_dir
                        backtrack_type = 2
                        point_path = point_path[:-1] # 删掉最后一步
                        dir_path = dir_path[:-1] # 删掉最后一步
                        continue # 回到while true循环
            
                if next_pos in point_path: # 交叉了，但不是一个loop（因为如果是loop，就会进入前面一个if分支）
                    cur_dir_last_try = next_dir
                    backtrack_type = 2
                    point_path = point_path[:-1] # 删掉最后一步
                    dir_path = dir_path[:-1] # 删掉最后一步
                    continue # 回到while true循环

            else: # last_flag == 2, 转一圈转回来了
                cur_dir_last_try = dir_path[-1] # 可能放if里第一行？
                point_path = point_path[:-1]
                dir_path = dir_path[:-1]
                backtrack_type = 2
        

if __name__ == '__main__':
    overall_limit = '22****3*223**23***3*3****'
    problem = slitherlink(5, 5, constraint=np.array(list(overall_limit)).reshape(5, 5))
    # problem = generate_problem_from_url(url='http://www.puzzle-loop.com/?v=0&size=4')
    problem.print_problem()
    if backtracking_solve(problem):
        print('success')
    # problem.print_solution()
