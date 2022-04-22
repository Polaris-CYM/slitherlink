import time
from sqlalchemy import true
from z3 import *
from problem_define import slitherlink
from generate_problem import *
from backjumping import *

def find_digit_neighbor(nrow, ncol, digit_row, digit_col):
    '''返回digit数字周围的四个边的坐标'''
    rs1 = (digit_row, digit_col)
    rs2 = (digit_row+1, digit_col)
    cs1 = (digit_row, digit_col)
    cs2 = (digit_row, digit_col+1)
    return rs1, rs2, cs1, cs2


def add_constraint_digit(s, problem, rs_list, cs_list):
    '''根据数字添加约束'''
    nrow, ncol = problem.nrow, problem.ncol
    remain = problem.constraint

    for row in range(nrow):
        for col in range(ncol):
            # 分析每一个数字
            digit = remain[row, col]
            if digit == '3':
                # 周围4个有3个是True，1个是False
                rs1, rs2, cs1, cs2 = find_digit_neighbor(nrow, ncol, row, col)
                # 4选3
                case1 = And(rs_list[rs1], rs_list[rs2], cs_list[cs1], Not(cs_list[cs2]))
                case2 = And(rs_list[rs1], rs_list[rs2], Not(cs_list[cs1]), cs_list[cs2])
                case3 = And(rs_list[rs1], Not(rs_list[rs2]), cs_list[cs1], cs_list[cs2])
                case4 = And(Not(rs_list[rs1]), rs_list[rs2], cs_list[cs1], cs_list[cs2])
                s.add(Or(case1, case2, case3, case4))
            elif digit == '2':
                # 周围4个有2个是True，2个是False
                rs1, rs2, cs1, cs2 = find_digit_neighbor(nrow, ncol, row, col)
                # 4选2
                case1 = And(rs_list[rs1], rs_list[rs2], Not(cs_list[cs1]), Not(cs_list[cs2]))
                case2 = And(rs_list[rs1], Not(rs_list[rs2]), cs_list[cs1], Not(cs_list[cs2]))
                case3 = And(rs_list[rs1], Not(rs_list[rs2]), Not(cs_list[cs1]), cs_list[cs2])
                case4 = And(Not(rs_list[rs1]), rs_list[rs2], cs_list[cs1], Not(cs_list[cs2]))
                case5 = And(Not(rs_list[rs1]), rs_list[rs2], Not(cs_list[cs1]), cs_list[cs2])
                case6 = And(Not(rs_list[rs1]), Not(rs_list[rs2]), cs_list[cs1], cs_list[cs2])
                s.add(Or(case1, case2, case3, case4, case5, case6))
            elif digit == '1':
                # 周围4个有1个是True，3个是False
                rs1, rs2, cs1, cs2 = find_digit_neighbor(nrow, ncol, row, col)
                # 4选3
                case1 = And(rs_list[rs1], Not(rs_list[rs2]), Not(cs_list[cs1]), Not(cs_list[cs2]))
                case2 = And(Not(rs_list[rs1]), rs_list[rs2], Not(cs_list[cs1]), Not(cs_list[cs2]))
                case3 = And(Not(rs_list[rs1]), Not(rs_list[rs2]), cs_list[cs1], Not(cs_list[cs2]))
                case4 = And(Not(rs_list[rs1]), Not(rs_list[rs2]), Not(cs_list[cs1]), cs_list[cs2])
                s.add(Or(case1, case2, case3, case4))
            elif digit == '0':
                rs1, rs2, cs1, cs2 = find_digit_neighbor(nrow, ncol, row, col)
                # 四个都全为0
                s.add(And(Not(rs_list[rs1]), Not(rs_list[rs2]),Not(cs_list[cs1]), Not(cs_list[cs2])))
            # else: # digit == '*'
            #     continue  # 不需要操作


def find_left_neighbor(nrow, ncol, row, col):
    '''
    找横边的左边的1/2/3条边
    (row, col)是横边的坐标
    '''
    if row == 0 and col == 0: # 左上角那条横边
        left1 = (0,0) # 竖边
        return [left1], [] # 竖边，横边
    elif row == nrow and col == 0: # 左下角
        left1 = (nrow-1,0)
        return [left1], []
    elif col == 0: # 靠左的横边（非最上最下）
        left1 = (row-1,0) # 竖边
        left2 = (row,0) # 竖边
        return [left1, left2], []
    elif row == 0: # 最上一行（非最左最右）
        left1 = (0, col) # 竖边
        left2 = (0, col-1) # 横边
        return [left1], [left2]
    elif row == nrow:
        left1 = (nrow-1, col) # 竖边
        left2 = (row, col-1) # 横边
        return [left1], [left2]
    else: # 其他边
        left1 = (row-1,col) # 竖边
        left2 = (row,col) # 竖边
        left3 = (row,col-1) # 横边
        return [left1, left2], [left3]


def find_right_neighbor(nrow, ncol, row, col):
    '''
    找横边的右边的1/2/3条边
    (row, col)是横边的坐标
    '''
    if row == 0 and col == ncol-1: # 右上角那条横边
        right1 = (0,ncol)
        return [right1], [] # 竖边，横边
    elif row == nrow and col == ncol-1: # 右下角
        right1 = (nrow-1,ncol)
        return [right1], []
    elif col == ncol-1: # 靠右的横边（非最上最下）
        right1 = (row-1,ncol)
        right2 = (row,ncol)
        return [right1, right2],[]
    elif row == 0: # 最上一行（非最左最右）
        right1 = (0, col+1) # 竖边
        right2 = (0, col+1) # 横边
        return [right1], [right2]
    elif row == nrow: # 最下一行
        right1 = (nrow-1, col+1) # 竖边
        right2 = (nrow, col+1)  # 横边
        return [right1], [right2]
    else: # 其他边
        right1 = (row-1,col+1) # 竖边
        right2 = (row,col+1) # 竖边
        right3 = (row,col+1) # 横边
        return [right1, right2], [right3]


def find_up_neighbor(nrow, ncol, row, col):
    '''
    找竖边的上面的1/2/3条边，(row, col)是竖边的坐标
    '''
    if row == 0 and col == 0: # 左上角
        return [], [(0,0)] # 竖边，横边
    elif row == 0 and col == ncol: # 右上角
        return [], [(0,ncol-1)]
    elif row == 0: # 靠上的边
        return [], [(0,col-1), (0,col)]
    elif col == 0: # 最左一列
        return [(row-1,0)], [(row,0)] # 竖边，横边
    elif col == ncol: # 最右一列
        return [(row-1,col)], [(row,col-1)] # 竖边，横边
    else:
        return [(row-1,col)], [(row,col-1), (row,col)] # 竖边，横边

def find_down_neighbor(nrow, ncol, row, col):
    '''
    找竖边的下面的1/2/3条边，(row, col)是竖边的坐标
    '''
    if row == nrow-1 and col == 0: # 左下角
        return [], [(nrow,0)] # 竖边，横边
    elif row == nrow-1 and col == ncol: # 右下角
        return [], [(nrow,ncol-1)]
    elif row == nrow-1: # 靠下的边
        return [], [(nrow,col-1), (nrow,col)]
    elif col == 0: # 最左一列
        return [(row+1,0)], [(row+1,0)] # 竖边，横边
    elif col == ncol: # 最右一列
        return [(row+1,col)], [(row+1,col-1)] # 竖边，横边
    else:
        return [(row+1,col)], [(row+1,col-1), (row+1,col)] # 竖边，横边


def add_constraint_naive(s, nrow, ncol, rs_list, cs_list):
    '''添加构成环的约束'''
    for row in range(nrow+1):
        for col in range(ncol):
            # 对row_solution[row, col]进行分析
            # 它应当满足与左边的1/2/3条边之一相连
            # 且与右边的1/2/3条边之一相连
            left_shu, left_heng = find_left_neighbor(nrow, ncol, row, col)
            num_left_shu, num_left_heng = len(left_shu), len(left_heng) # num_shu=1/2, num_heng=0/1
            if num_left_shu == 1 and num_left_heng == 0:
                left1 = left_shu[0]
                case1 = Not(rs_list[row,col]) # 要么这条横线不是解
                case2 = cs_list[left1] # 要么它相邻的边要是解
                s.add(Or(case1, case2))
            elif num_left_shu == 1 and num_left_heng == 1:
                left1, left2 = left_shu[0], left_heng[0]
                case1 = Not(rs_list[row,col]) # 要么这条横线不是解
                case2 = And(cs_list[left1], Not(rs_list[left2])) # 要么left1是解
                case3 = And(Not(cs_list[left1]), rs_list[left2]) # 要么left2是解
                s.add(Or(case1, case2, case3))
            elif num_left_shu == 2 and num_left_heng == 0:
                left1, left2 = left_shu[0], left_shu[1]
                case1 = Not(rs_list[row,col]) # 要么这条横线不是解
                case2 = And(cs_list[left1], Not(cs_list[left2])) # 要么left1是解
                case3 = And(Not(cs_list[left1]), cs_list[left2]) # 要么left2是解
                s.add(Or(case1, case2, case3))
            elif num_left_shu == 2 and num_left_heng == 1:
                left1, left2, left3 = left_shu[0], left_shu[1], left_heng[0]
                case1 = Not(rs_list[row,col]) # 要么这条横线不是解
                case2 = And(cs_list[left1], Not(cs_list[left2]), Not(rs_list[left3])) # 要么left1是解
                case3 = And(Not(cs_list[left1]), cs_list[left2], Not(rs_list[left3])) # 要么left2是解
                case4 = And(Not(cs_list[left1]), Not(cs_list[left2]), rs_list[left3]) # 要么left3是解
                s.add(Or(case1, case2, case3, case4))

            right_shu, right_heng = find_right_neighbor(nrow, ncol, row, col)
            num_right_shu, num_right_heng = len(right_shu), len(right_heng) # num_shu=1/2, num_heng=0/1
            if num_right_shu == 1 and num_right_heng == 0:
                right1 = right_shu[0]
                case1 = Not(rs_list[row,col]) # 要么这条横线不是解
                case2 = cs_list[right1] # 要么它相邻的边要是解
                s.add(Or(case1, case2))
            elif num_right_shu == 1 and num_right_heng == 1:
                right1, right2 = right_shu[0], right_heng[0]
                case1 = Not(rs_list[row,col]) # 要么这条横线不是解
                case2 = And(cs_list[right1], Not(rs_list[right2])) # 要么right1是解
                case3 = And(Not(cs_list[right1]), rs_list[right2]) # 要么right2是解
                s.add(Or(case1, case2, case3))
            elif num_right_shu == 2 and num_right_heng == 0:
                right1, right2 = right_shu[0], right_shu[1]
                case1 = Not(rs_list[row,col]) # 要么这条横线不是解
                case2 = And(cs_list[right1], Not(cs_list[right2])) # 要么right1是解
                case3 = And(Not(cs_list[right1]), cs_list[right2]) # 要么right2是解
                s.add(Or(case1, case2, case3))
            elif num_right_shu == 2 and num_right_heng == 1:
                right1, right2, right3 = right_shu[0], right_shu[1], right_heng[0]
                case1 = Not(rs_list[row,col]) # 要么这条横线不是解
                case2 = And(cs_list[right1], Not(cs_list[right2]), Not(rs_list[right3])) # 要么right1是解
                case3 = And(Not(cs_list[right1]), cs_list[right2], Not(rs_list[right3])) # 要么right2是解
                case4 = And(Not(cs_list[right1]), Not(cs_list[right2]), rs_list[right3]) # 要么right3是解
                s.add(Or(case1, case2, case3, case4))

    for row in range(nrow):
        for col in range(ncol+1):
            # 对col_solution[row, col]进行分析
            # 它应当满足与上边的1/2/3条边之一相连
            # 且与下边的1/2/3条边之一相连
            up_shu, up_heng = find_up_neighbor(nrow, ncol, row, col)
            num_up_shu, num_up_heng = len(up_shu), len(up_heng) # num_shu=0/1, num_heng=1/2
            if num_up_shu == 0 and num_up_heng == 1:
                up1 = up_heng[0]
                case1 = Not(cs_list[row,col])
                case2 = rs_list[up1]
                s.add(Or(case1, case2))
            elif num_up_shu == 0 and num_up_heng == 2:
                up1, up2 = up_heng[0], up_heng[1]
                case1 = Not(cs_list[row,col])
                case2 = And(rs_list[up1], Not(rs_list[up2]))
                case3 = And(Not(rs_list[up1]), rs_list[up2])
                s.add(Or(case1, case2, case3))
            elif num_up_shu == 1 and num_up_heng == 1:
                up1, up2 = up_shu[0], up_heng[0]
                case1 = Not(cs_list[row,col])
                case2 = And(cs_list[up1], Not(rs_list[up2]))
                case3 = And(Not(cs_list[up1]), rs_list[up2])
                s.add(Or(case1, case2, case3))
            elif num_up_shu == 1 and num_up_heng == 2:
                up1, up2, up3 = up_shu[0], up_heng[0], up_heng[1]
                case1 = Not(cs_list[row,col])
                case2 = And(cs_list[up1], Not(rs_list[up2]), Not(rs_list[up3]))
                case3 = And(Not(cs_list[up1]), rs_list[up2], Not(rs_list[up3]))
                case4 = And(Not(cs_list[up1]), Not(rs_list[up2]), rs_list[up3])
                s.add(Or(case1, case2, case3, case4))

            down_shu, down_heng = find_down_neighbor(nrow, ncol, row, col)
            num_down_shu, num_down_heng = len(down_shu), len(down_heng) # num_shu=0/1, num_heng=1/2
            if num_down_shu == 0 and num_down_heng == 1:
                down1 = down_heng[0]
                case1 = Not(cs_list[row,col])
                case2 = rs_list[down1]
                s.add(Or(case1, case2))
            elif num_down_shu == 0 and num_down_heng == 2:
                down1, down2 = down_heng[0], down_heng[1]
                case1 = Not(cs_list[row,col])
                case2 = And(rs_list[down1], Not(rs_list[down2]))
                case3 = And(Not(rs_list[down1]), rs_list[down2])
                s.add(Or(case1, case2, case3))
            elif num_down_shu == 1 and num_down_heng == 1:
                down1, down2 = down_shu[0], down_heng[0]
                case1 = Not(cs_list[row,col])
                case2 = And(cs_list[down1], Not(rs_list[down2]))
                case3 = And(Not(cs_list[down1]), rs_list[down2])
                s.add(Or(case1, case2, case3))
            elif num_down_shu == 1 and num_down_heng == 2:
                down1, down2, down3 = down_shu[0], down_heng[0], down_heng[1]
                case1 = Not(cs_list[row,col])
                case2 = And(cs_list[down1], Not(rs_list[down2]), Not(rs_list[down3]))
                case3 = And(Not(cs_list[down1]), rs_list[down2], Not(rs_list[down3]))
                case4 = And(Not(cs_list[down1]), Not(rs_list[down2]), rs_list[down3])
                s.add(Or(case1, case2, case3, case4))


def add_constraint_forbid_force(s, problem, rs_list, cs_list):
    nrow, ncol = problem.nrow, problem.ncol
    remain = problem.constraint.copy()
    row_forbid, col_forbid = find_forbid(remain)
    row_force, col_force = find_force(remain)

    for row_pos in row_forbid:
        s.add(Not(rs_list[row_pos]))
    for col_pos in col_forbid:
        s.add(Not(cs_list[col_pos]))
    for row_pos in row_force:
        s.add(rs_list[row_pos])
    for col_pos in col_force:
        s.add(cs_list[col_pos])


def is_legal_solution(problem):
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
    
    # 第一步的坐标
    first_pos = (start_row[0], start_col[0])
    next_pos = (end_row[0], end_col[0])
    # 每一条边是否还能被选
    can_select = [True] * edge_count
    can_select[0] = False # 第一条边已经选了
    # 实际构成环的边数
    count = 1

    while True:
        for idx in range(edge_count):
            if can_select[idx] and start_row[idx] == next_pos[0] and start_col[idx] == next_pos[1]:
                # 如果后面哪条边的start坐标能接上前一步的next_pos坐标
                next_pos = (end_row[idx], end_col[idx]) # 更新next_pos
                count += 1
                can_select[idx] = False
                break
            if can_select[idx] and end_row[idx] == next_pos[0] and end_col[idx] == next_pos[1]:
                # 如果后面哪条边的end坐标能接上前一步的next_pos坐标
                next_pos = (start_row[idx], start_col[idx]) # 更新next_pos
                count += 1
                can_select[idx] = False
                break
        if next_pos[0]==first_pos[0] and next_pos[1] == first_pos[1]:
            # 如果构成了一个环，那么就跳出while循环
            break # 跳出while循环
    
    if count == edge_count:
        return True
    else:
        return False

def output_solution(s_model, problem, rs_list, cs_list):
    nrow, ncol = problem.nrow, problem.ncol
    # 初始化，归零
    problem.row_solution = np.zeros(shape=(nrow+1, ncol))
    problem.col_solution = np.zeros(shape=(nrow, ncol+1))
    # 根据s_model赋值
    for row in range(nrow+1):
        for col in range(ncol):
            if s_model.eval(rs_list[row,col]):
                problem.row_solution[row,col] = 1
    for row in range(nrow):
        for col in range(ncol+1):
            if s_model.eval(cs_list[row,col]):
                problem.col_solution[row,col] = 1 
    
    if is_legal_solution(problem):
        # output solution
        print()
        print("SAT SOLUTION:")
        problem.print_solution()
        return True
    else:
        return False


def sat_solve(problem):
    nrow, ncol = problem.nrow, problem.ncol

    rs_name_list = []  # ['rs_0_0', 'rs_0_1', 'rs_0_2', ... ]
    for row in range(nrow+1):
        for col in range(ncol):
            rs_name_list.append('rs_'+str(row)+'_'+str(col))

    cs_name_list = []  # ['cs_0_0', 'cs_0_1', 'cs_0_2', ... ]
    for row in range(nrow):
        for col in range(ncol+1):
            cs_name_list.append('cs_'+str(row)+'_'+str(col))

    rs_list = np.array([Bool(rs_name) for rs_name in rs_name_list]).reshape((nrow+1, ncol))
    cs_list = np.array([Bool(cs_name) for cs_name in cs_name_list]).reshape((nrow, ncol+1))

    s = Solver()
    add_constraint_forbid_force(s, problem, rs_list, cs_list)  # 增加必选必不选
    add_constraint_digit(s, problem, rs_list, cs_list)  # 根据数字添加约束
    add_constraint_naive(s, nrow, ncol, rs_list, cs_list)  # 添加构成环的约束
    result = []

    # 验证多个解
    while s.check() == z3.sat:
        # https://stackoverflow.com/questions/11867611/z3py-checking-all-solutions-for-equation
        m = s.model()
        if output_solution(m, problem, rs_list, cs_list):
            # 如果已经产生合法解，就退出
            return True
        result.append(m)
        block = []
        for d in m:
            # d is a declaration
            if d.arity() > 0:
                raise Z3Exception("uninterpreted functions are not supported")
            # create a constant from declaration
            c = d()
            if is_array(c) or c.sort().kind() == Z3_UNINTERPRETED_SORT:
                raise Z3Exception("arrays and uninterpreted sorts are not supported")
            block.append(c != m[d])
        s.add(Or(block))

    return False

if __name__ == '__main__':
    # overall_limit = '222333'  #2*3
    # overall_limit = '323*221**3233213'
    # nrow, ncol = 4,4
    # problem = slitherlink(nrow, ncol, constraint=np.array(list(overall_limit)).reshape(nrow, ncol))
    problem = generate_problem_from_url(url='http://www.puzzle-loop.com/?v=0&size=0')
    problem.print_problem()

    start_time = time.time()
    sat_solve(problem)
    end_time = time.time()
    print()
    print('start time: {}'.format(start_time))
    print('end time: {}'.format(end_time))
    print('time cost: {}'.format(end_time-start_time))
