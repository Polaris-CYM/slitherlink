import time
import numpy as np
# from pyparsing import col
from problem_define import slitherlink
from generate_problem import *
from backtracking import *


def row_solution_pos_legal(pos, nrow, ncol):
    '''
    pos should within [0 ... nrow, 0 ... ncol-1]
    '''
    if 0 <= pos[0] <= nrow and 0<= pos[1] <= ncol - 1:
        return True
    else:
        return False


def col_solution_pos_legal(pos, nrow, ncol):
    '''
    pos should within [0 ... nrow-1, 0 ... ncol]
    '''
    if 0 <= pos[0] <= nrow - 1 and 0<= pos[1] <= ncol:
        return True
    else:
        return False


def find_forbid(remain):
    nrow, ncol = remain.shape
    row_forbid = [] # row_forbid = [(r1,c1), (r2,c2), ...] 表示problem.row_solution(ri,ci)不能成为解
    col_forbid = [] # col_forbid = [(r1,c1), (r2,c2), ...] 表示problem.col_solution(ri,ci)不能成为解

    # situation 1: 1在角落
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

    # situation 2: 33相连
    for row in range(nrow): # 从第一行digit到最后一行digit
        for col in range(ncol-1): # 从第一列数字到倒数第二列数字
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
    

def find_force(remain):
    nrow, ncol = remain.shape
    row_force = [] # row_force = [(r1,c1), (r2,c2), ...] 表示problem.row_solution(ri,ci)必须成为解
    col_force = [] # col_force = [(r1,c1), (r2,c2), ...] 表示problem.col_solution(ri,ci)必须成为解

    # situation 1: 3在角落
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
    
    # situation 2: 2在角落
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
    # situation 3: 30相邻
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
    
    return row_force, col_force


def forbid_allow(current_pos, next_dir, row_forbid, col_forbid):
    if next_dir == 'right':
        return False if current_pos in row_forbid else True
    if next_dir == 'left':
        return False if (current_pos[0],current_pos[1]-1) in row_forbid else True
    if next_dir == 'down':
        return False if current_pos in col_forbid else True
    if next_dir == 'up':
        return False if (current_pos[0]-1,current_pos[1]) in col_forbid else True

def construct_jump_solution(problem, point_path):
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
    print('BACKJUMPING SOLUTION:')
    problem.print_solution()
    return True

def backjumping_solve(problem):
    '''
    use the backtracking search algorithm to solve the problem
    '''
    nrow, ncol = problem.nrow, problem.ncol
    start_row, start_col = find_start(problem)  # 从这个点开始画圈，因为旁边就有3，所以画的第一条边大概率是解的一部分
    point_path = [(start_row, start_col)]  # 维护loop solution的点集合，点是square的顶点
    dir_path = []  # 从start这个点开始的方向
    remain = problem.constraint.copy()  # 维护一个constraint的副本，随时增减
    row_forbid, col_forbid = find_forbid(remain)
    current_pos = (start_row, start_col)  # 当前位置
    
    # 先向下走，向下画loop
    next_dir = 'down'
    if (not remain_allow(remain, current_pos, next_dir)) or (not forbid_allow(current_pos, next_dir, row_forbid, col_forbid)): 
        # 如果不能向下走（只可能因为03连一起）就向右边走   或者   如果这么走在forbid名单里
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

            while (not remain_allow(remain, current_pos, next_dir)) or (not forbid_allow(current_pos, next_dir, row_forbid, col_forbid)): 
                # 只有当 remain/界面大小 不允许的情况才会循环
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
                        construct_jump_solution(problem, point_path)
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
                while (not remain_allow(remain, current_pos, next_dir)) or (not forbid_allow(current_pos, next_dir, row_forbid, col_forbid)):
                    # 只有当 remain/界面大小 不允许的情况才会循环
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
                        construct_jump_solution(problem, point_path)
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


