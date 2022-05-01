import time
import numpy as np
from problem_define import slitherlink
from generate_problem import *
from constraint_propagation_forward import *


def is_legal_solution(problem):
    '''
    区别于sat_solve里面的is_legal_solution，
    这里的is_legal_solution多了一个update状态。
    sat_solve里面一定是环，需要判断是否是多个环
    这里甚至都不能组成环
    '''

    # 每一个解里面的所有边的point的坐标值
    # 例如第一条横边的四个值是 0001，从point(0,0)到point(0,1)
    start_row, start_col, end_row, end_col = [], [], [], []
    edge_count = 0 # 有多少条被选出来的边的数量，可能有多个环的边的数量
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
    
    num_sum = np.sum(problem.constraint=='1') + 2*np.sum(problem.constraint=='2') + 3*np.sum(problem.constraint=='3')
    if 2*edge_count < num_sum:
        return False

    # 第一步的坐标
    first_pos = (start_row[0], start_col[0])
    next_pos = (end_row[0], end_col[0])
    # 每一条边是否还能被选
    can_select = [True] * edge_count
    can_select[0] = False # 第一条边已经选了
    # 实际构成环的边数
    count = 1
    update = True
    while update:
        update = False
        for idx in range(edge_count):
            if can_select[idx] and start_row[idx] == next_pos[0] and start_col[idx] == next_pos[1]:
                # 如果后面哪条边的start坐标能接上前一步的next_pos坐标
                next_pos = (end_row[idx], end_col[idx]) # 更新next_pos
                count += 1
                can_select[idx] = False
                update = True
                break
            if can_select[idx] and end_row[idx] == next_pos[0] and end_col[idx] == next_pos[1]:
                # 如果后面哪条边的end坐标能接上前一步的next_pos坐标
                next_pos = (start_row[idx], start_col[idx]) # 更新next_pos
                count += 1
                can_select[idx] = False
                update = True
                break
        if next_pos[0]==first_pos[0] and next_pos[1] == first_pos[1]:
            # 如果构成了一个环，那么就跳出while循环
            break # 跳出while循环
    
    if count == edge_count:
        return True
    else:
        return False


def naive_constraint_propagation(problem):
    remain = problem.constraint.copy()
    # row_forbid, col_forbid = find_forbid(remain)
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
    # problem.print_solution()
