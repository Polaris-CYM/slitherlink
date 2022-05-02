import time
# from sqlalchemy import true
from z3 import *
from problem_define import slitherlink
from generate_problem import *
from constraint_propagation_forward import *


def find_digit_neighbor(nrow, ncol, digit_row, digit_col):
    """
    Return the coordinates of the four edges around the digit
    """
    rs1 = (digit_row, digit_col)
    rs2 = (digit_row+1, digit_col)
    cs1 = (digit_row, digit_col)
    cs2 = (digit_row, digit_col+1)
    return rs1, rs2, cs1, cs2


def add_constraint_digit(s, problem, rs_list, cs_list):
    """
    Add numeric constraints
    """
    nrow, ncol = problem.nrow, problem.ncol
    remain = problem.constraint

    for row in range(nrow):
        for col in range(ncol):
            digit = remain[row, col]
            if digit == '3':
                rs1, rs2, cs1, cs2 = find_digit_neighbor(nrow, ncol, row, col)
                # 3 of the 4 surrounding edges are True and 1 is False
                case1 = And(rs_list[rs1], rs_list[rs2], cs_list[cs1], Not(cs_list[cs2]))
                case2 = And(rs_list[rs1], rs_list[rs2], Not(cs_list[cs1]), cs_list[cs2])
                case3 = And(rs_list[rs1], Not(rs_list[rs2]), cs_list[cs1], cs_list[cs2])
                case4 = And(Not(rs_list[rs1]), rs_list[rs2], cs_list[cs1], cs_list[cs2])
                s.add(Or(case1, case2, case3, case4))
            elif digit == '2':
                rs1, rs2, cs1, cs2 = find_digit_neighbor(nrow, ncol, row, col)
                # 2 of the 4 surrounding edges are True and 2 are False
                case1 = And(rs_list[rs1], rs_list[rs2], Not(cs_list[cs1]), Not(cs_list[cs2]))
                case2 = And(rs_list[rs1], Not(rs_list[rs2]), cs_list[cs1], Not(cs_list[cs2]))
                case3 = And(rs_list[rs1], Not(rs_list[rs2]), Not(cs_list[cs1]), cs_list[cs2])
                case4 = And(Not(rs_list[rs1]), rs_list[rs2], cs_list[cs1], Not(cs_list[cs2]))
                case5 = And(Not(rs_list[rs1]), rs_list[rs2], Not(cs_list[cs1]), cs_list[cs2])
                case6 = And(Not(rs_list[rs1]), Not(rs_list[rs2]), cs_list[cs1], cs_list[cs2])
                s.add(Or(case1, case2, case3, case4, case5, case6))
            elif digit == '1':
                rs1, rs2, cs1, cs2 = find_digit_neighbor(nrow, ncol, row, col)
                # 1 of the 4 surrounding edges is True and 3 are False
                case1 = And(rs_list[rs1], Not(rs_list[rs2]), Not(cs_list[cs1]), Not(cs_list[cs2]))
                case2 = And(Not(rs_list[rs1]), rs_list[rs2], Not(cs_list[cs1]), Not(cs_list[cs2]))
                case3 = And(Not(rs_list[rs1]), Not(rs_list[rs2]), cs_list[cs1], Not(cs_list[cs2]))
                case4 = And(Not(rs_list[rs1]), Not(rs_list[rs2]), Not(cs_list[cs1]), cs_list[cs2])
                s.add(Or(case1, case2, case3, case4))
            elif digit == '0':
                rs1, rs2, cs1, cs2 = find_digit_neighbor(nrow, ncol, row, col)
                # All 4 edges are False
                s.add(And(Not(rs_list[rs1]), Not(rs_list[rs2]),Not(cs_list[cs1]), Not(cs_list[cs2])))
            # else: # digit == '*'
            #     continue


def find_left_neighbor(nrow, ncol, row, col):
    """
    Find the edge(s) adjacent to the left endpoint of the horizontal edge (there may be 1/2/3 edges)
    (row, col) is the coordinate of the target horizontal edge
    """
    if row == 0 and col == 0:  # The horizontal edge in the upper left corner
        left1 = (0,0)
        return [left1], []  # [vertical edge(s)], [horizontal edge(s)]
    elif row == nrow and col == 0:  # The horizontal edge in the lower left corner
        left1 = (nrow-1,0)
        return [left1], []
    elif col == 0:  # Horizontal edges intersecting with the left border (not in the corner)
        left1 = (row-1,0)
        left2 = (row,0)
        return [left1, left2], []
    elif row == 0:  # on the first row
        left1 = (0, col)
        left2 = (0, col-1)
        return [left1], [left2]
    elif row == nrow:  # on the last row
        left1 = (nrow-1, col)
        left2 = (row, col-1)
        return [left1], [left2]
    else:
        left1 = (row-1,col)
        left2 = (row,col)
        left3 = (row,col-1)
        return [left1, left2], [left3]


def find_right_neighbor(nrow, ncol, row, col):
    """
    Find the edge(s) adjacent to the right endpoint of the horizontal edge (there may be 1/2/3 edges)
    (row, col) is the coordinate of the target horizontal edge
    """
    if row == 0 and col == ncol-1:  # The horizontal edge in the upper right corner
        right1 = (0,ncol)
        return [right1], []  # [vertical edge(s)], [horizontal edge(s)]
    elif row == nrow and col == ncol-1:  # The horizontal edge in the lower right corner
        right1 = (nrow-1,ncol)
        return [right1], []
    elif col == ncol-1:  # Horizontal edges intersecting with the right border (not in the corner)
        right1 = (row-1,ncol)
        right2 = (row,ncol)
        return [right1, right2],[]
    elif row == 0:  # on the first row
        right1 = (0, col+1)
        right2 = (0, col+1)
        return [right1], [right2]
    elif row == nrow:  # on the last row
        right1 = (nrow-1, col+1)
        right2 = (nrow, col+1)
        return [right1], [right2]
    else:
        right1 = (row-1,col+1)
        right2 = (row,col+1)
        right3 = (row,col+1)
        return [right1, right2], [right3]


def find_up_neighbor(nrow, ncol, row, col):
    """
    Find the edge(s) adjacent to the upper endpoint of the vertical edge (there may be 1/2/3 edges)
    (row, col) is the coordinate of the target vertical edge
    """
    if row == 0 and col == 0:  # The vertical edge in the upper left corner
        return [], [(0,0)]  # [vertical edge(s)], [horizontal edge(s)]
    elif row == 0 and col == ncol:  # The vertical edge in the upper right corner
        return [], [(0,ncol-1)]
    elif row == 0:  # Vertical edges intersecting the upper boundary (not in the corner)
        return [], [(0,col-1), (0,col)]
    elif col == 0:  # on the leftmost column
        return [(row-1,0)], [(row,0)]
    elif col == ncol:  # on the rightmost column
        return [(row-1,col)], [(row,col-1)]
    else:
        return [(row-1,col)], [(row,col-1), (row,col)]


def find_down_neighbor(nrow, ncol, row, col):
    """
    Find the edge(s) adjacent to the lower endpoint of the vertical edge (there may be 1/2/3 edges)
    (row, col) is the coordinate of the target vertical edge
    """
    if row == nrow-1 and col == 0: # The vertical edge in the lower left corner
        return [], [(nrow,0)]
    elif row == nrow-1 and col == ncol:  # The vertical edge in the lower right corner
        return [], [(nrow,ncol-1)]
    elif row == nrow-1:  # Vertical edges intersecting the lower boundary (not in the corner)
        return [], [(nrow,col-1), (nrow,col)]
    elif col == 0:  # on the leftmost column
        return [(row+1,0)], [(row+1,0)]
    elif col == ncol:  # on the rightmost column
        return [(row+1,col)], [(row+1,col-1)]
    else:
        return [(row+1,col)], [(row+1,col-1), (row+1,col)]


def add_constraint_naive(s, nrow, ncol, rs_list, cs_list):
    """
    Add constraints that form a loop
    """
    for row in range(nrow+1):
        for col in range(ncol):
            # For each row_solution[row, col], it should satisfy:
            # The left endpoint is connected to one of the adjacent 1/2/3 edge(s),
            # and the right endpoint is connected to one of the adjacent 1/2/3 edge(s)
            left_vertical, left_horizontal = find_left_neighbor(nrow, ncol, row, col)
            num_left_v, num_left_h = len(left_vertical), len(left_horizontal)  # num_left_v=1 or 2, num_left_h=0 or 1
            if num_left_v == 1 and num_left_h == 0:
                left1 = left_vertical[0]
                case1 = Not(rs_list[row,col])  # The target horizontal line is not in the solution
                case2 = cs_list[left1]  # The only edge adjacent to its left endpoint must also be in the solution
                s.add(Or(case1, case2))
            elif num_left_v == 1 and num_left_h == 1:
                left1, left2 = left_vertical[0], left_horizontal[0]
                case1 = Not(rs_list[row,col])
                case2 = And(cs_list[left1], Not(rs_list[left2]))  # left1 is in the solution
                case3 = And(Not(cs_list[left1]), rs_list[left2])  # left2 is in the solution
                s.add(Or(case1, case2, case3))
            elif num_left_v == 2 and num_left_h == 0:
                left1, left2 = left_vertical[0], left_vertical[1]
                case1 = Not(rs_list[row,col])
                case2 = And(cs_list[left1], Not(cs_list[left2]))
                case3 = And(Not(cs_list[left1]), cs_list[left2])
                s.add(Or(case1, case2, case3))
            elif num_left_v == 2 and num_left_h == 1:
                left1, left2, left3 = left_vertical[0], left_vertical[1], left_horizontal[0]
                case1 = Not(rs_list[row,col])
                case2 = And(cs_list[left1], Not(cs_list[left2]), Not(rs_list[left3]))
                case3 = And(Not(cs_list[left1]), cs_list[left2], Not(rs_list[left3]))
                case4 = And(Not(cs_list[left1]), Not(cs_list[left2]), rs_list[left3])
                s.add(Or(case1, case2, case3, case4))

            right_vertical, right_horizontal = find_right_neighbor(nrow, ncol, row, col)
            num_right_v, num_right_h = len(right_vertical), len(right_horizontal)
            if num_right_v == 1 and num_right_h == 0:
                right1 = right_vertical[0]
                case1 = Not(rs_list[row,col])
                case2 = cs_list[right1]
                s.add(Or(case1, case2))
            elif num_right_v == 1 and num_right_h == 1:
                right1, right2 = right_vertical[0], right_horizontal[0]
                case1 = Not(rs_list[row,col])
                case2 = And(cs_list[right1], Not(rs_list[right2]))
                case3 = And(Not(cs_list[right1]), rs_list[right2])
                s.add(Or(case1, case2, case3))
            elif num_right_v == 2 and num_right_h == 0:
                right1, right2 = right_vertical[0], right_vertical[1]
                case1 = Not(rs_list[row,col])
                case2 = And(cs_list[right1], Not(cs_list[right2]))
                case3 = And(Not(cs_list[right1]), cs_list[right2])
                s.add(Or(case1, case2, case3))
            elif num_right_v == 2 and num_right_h == 1:
                right1, right2, right3 = right_vertical[0], right_vertical[1], right_horizontal[0]
                case1 = Not(rs_list[row,col])
                case2 = And(cs_list[right1], Not(cs_list[right2]), Not(rs_list[right3]))
                case3 = And(Not(cs_list[right1]), cs_list[right2], Not(rs_list[right3]))
                case4 = And(Not(cs_list[right1]), Not(cs_list[right2]), rs_list[right3])
                s.add(Or(case1, case2, case3, case4))

    for row in range(nrow):
        for col in range(ncol+1):
            # For each col_solution[row, col], it should satisfy:
            # The upper endpoint is connected to one of the adjacent 1/2/3 edge(s),
            # and the lower endpoint is connected to one of the adjacent 1/2/3 edge(s)
            up_vertical, up_horizontal = find_up_neighbor(nrow, ncol, row, col)
            num_up_v, num_up_h = len(up_vertical), len(up_horizontal)
            if num_up_v == 0 and num_up_h == 1:
                up1 = up_horizontal[0]
                case1 = Not(cs_list[row,col])
                case2 = rs_list[up1]
                s.add(Or(case1, case2))
            elif num_up_v == 0 and num_up_h == 2:
                up1, up2 = up_horizontal[0], up_horizontal[1]
                case1 = Not(cs_list[row,col])
                case2 = And(rs_list[up1], Not(rs_list[up2]))
                case3 = And(Not(rs_list[up1]), rs_list[up2])
                s.add(Or(case1, case2, case3))
            elif num_up_v == 1 and num_up_h == 1:
                up1, up2 = up_vertical[0], up_horizontal[0]
                case1 = Not(cs_list[row,col])
                case2 = And(cs_list[up1], Not(rs_list[up2]))
                case3 = And(Not(cs_list[up1]), rs_list[up2])
                s.add(Or(case1, case2, case3))
            elif num_up_v == 1 and num_up_h == 2:
                up1, up2, up3 = up_vertical[0], up_horizontal[0], up_horizontal[1]
                case1 = Not(cs_list[row,col])
                case2 = And(cs_list[up1], Not(rs_list[up2]), Not(rs_list[up3]))
                case3 = And(Not(cs_list[up1]), rs_list[up2], Not(rs_list[up3]))
                case4 = And(Not(cs_list[up1]), Not(rs_list[up2]), rs_list[up3])
                s.add(Or(case1, case2, case3, case4))

            down_vertical, down_horizontal = find_down_neighbor(nrow, ncol, row, col)
            num_down_v, num_down_h = len(down_vertical), len(down_horizontal)
            if num_down_v == 0 and num_down_h == 1:
                down1 = down_horizontal[0]
                case1 = Not(cs_list[row,col])
                case2 = rs_list[down1]
                s.add(Or(case1, case2))
            elif num_down_v == 0 and num_down_h == 2:
                down1, down2 = down_horizontal[0], down_horizontal[1]
                case1 = Not(cs_list[row,col])
                case2 = And(rs_list[down1], Not(rs_list[down2]))
                case3 = And(Not(rs_list[down1]), rs_list[down2])
                s.add(Or(case1, case2, case3))
            elif num_down_v == 1 and num_down_h == 1:
                down1, down2 = down_vertical[0], down_horizontal[0]
                case1 = Not(cs_list[row,col])
                case2 = And(cs_list[down1], Not(rs_list[down2]))
                case3 = And(Not(cs_list[down1]), rs_list[down2])
                s.add(Or(case1, case2, case3))
            elif num_down_v == 1 and num_down_h == 2:
                down1, down2, down3 = down_vertical[0], down_horizontal[0], down_horizontal[1]
                case1 = Not(cs_list[row,col])
                case2 = And(cs_list[down1], Not(rs_list[down2]), Not(rs_list[down3]))
                case3 = And(Not(cs_list[down1]), rs_list[down2], Not(rs_list[down3]))
                case4 = And(Not(cs_list[down1]), Not(rs_list[down2]), rs_list[down3])
                s.add(Or(case1, case2, case3, case4))


def add_constraint_forbid_force(s, problem, rs_list, cs_list):
    """
    Add constraints for prohibited and mandatory edges
    """
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
    """
    Determine if only one loop is formed
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

    first_pos = (start_row[0], start_col[0])
    next_pos = (end_row[0], end_col[0])
    can_select = [True] * edge_count
    can_select[0] = False  # The first edge has been selected
    count = 1  # Number of edges forming the first loop

    while True:
        for idx in range(edge_count):
            if can_select[idx] and start_row[idx] == next_pos[0] and start_col[idx] == next_pos[1]:
                next_pos = (end_row[idx], end_col[idx])
                count += 1
                can_select[idx] = False
                break
            if can_select[idx] and end_row[idx] == next_pos[0] and end_col[idx] == next_pos[1]:
                next_pos = (start_row[idx], start_col[idx])
                count += 1
                can_select[idx] = False
                break
        # If a loop is formed, then jump out of the while loop
        if next_pos[0]==first_pos[0] and next_pos[1] == first_pos[1]:
            break

    if count == edge_count:  # The generated loop is the only loop in the solution
        return True
    else:
        return False


def output_solution(s_model, problem, rs_list, cs_list):
    nrow, ncol = problem.nrow, problem.ncol
    # Initialize to 0
    problem.row_solution = np.zeros(shape=(nrow+1, ncol))
    problem.col_solution = np.zeros(shape=(nrow, ncol+1))

    for row in range(nrow+1):
        for col in range(ncol):
            if s_model.eval(rs_list[row, col]):
                problem.row_solution[row, col] = 1
    for row in range(nrow):
        for col in range(ncol+1):
            if s_model.eval(cs_list[row, col]):
                problem.col_solution[row, col] = 1
    
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

    # Set the name for each horizontal edge ['rs_0_0', 'rs_0_1', 'rs_0_2', ... ]
    rs_name_list = []
    for row in range(nrow+1):
        for col in range(ncol):
            rs_name_list.append('rs_'+str(row)+'_'+str(col))

    # Set the name for each vertical edge ['cs_0_0', 'cs_0_1', 'cs_0_2', ... ]
    cs_name_list = []
    for row in range(nrow):
        for col in range(ncol+1):
            cs_name_list.append('cs_'+str(row)+'_'+str(col))

    rs_list = np.array([Bool(rs_name) for rs_name in rs_name_list]).reshape((nrow+1, ncol))
    cs_list = np.array([Bool(cs_name) for cs_name in cs_name_list]).reshape((nrow, ncol+1))

    s = Solver()  # Initialize a sat solver
    add_constraint_forbid_force(s, problem, rs_list, cs_list)
    add_constraint_digit(s, problem, rs_list, cs_list)
    add_constraint_naive(s, nrow, ncol, rs_list, cs_list)
    result = []

    while s.check() == z3.sat:  # If there exists a solution that satisfies all constraints
        m = s.model()
        if output_solution(m, problem, rs_list, cs_list):
            return True
        result.append(m)
        block = []
        # https://stackoverflow.com/questions/11867611/z3py-checking-all-solutions-for-equation
        for d in m:
            # d is a declaration
            if d.arity() > 0:
                raise Z3Exception("uninterpreted functions are not supported")
            # create a constant from declaration
            c = d()
            if is_array(c) or c.sort().kind() == Z3_UNINTERPRETED_SORT:
                raise Z3Exception("arrays and uninterpreted sorts are not supported")
            block.append(c != m[d])
        s.add(Or(block))  # Guarantee not to generate solutions that have been judged to be wrong

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
