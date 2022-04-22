import time
from generate_problem import *
from backjumping import *
from minizinc import Instance, Model, Solver


def constraint_to_remain(constraint):
    # constraint 是字符串的二维数组，要转化为list的形式
    nrow, ncol = constraint.shape
    remain = []
    for i in range(nrow):
        row_list = []
        for j in range(ncol):
            if constraint[i,j] == '3':
                row_list.append(3)
            elif constraint[i,j] == '2':
                row_list.append(2)
            elif constraint[i,j] == '1':
                row_list.append(1)
            elif constraint[i,j] == '0':
                row_list.append(0)
            else:
                row_list.append(-1)
        remain.append(row_list)
    return remain


def minizinc_solve(problem, solver_name):
    model = Model()

    # 所有题都要加这些约束，里面的nrow和ncol在后面传参数进去
    model.add_string("""
        int: nrow;
        int: ncol;
        array[1..nrow+1,1..ncol] of var 0..1: rs;
        array[1..nrow,1..ncol+1] of var 0..1: cs;
        array[1..nrow,1..ncol] of int: remain;

        % add digit constraint
        % constraint forall(i in digit_nrow_range, j in digit_ncol_range)(remain[i,j] = rs[i,j] + rs[i+1,j] + cs[i,j] + cs[i,j+1]);
        constraint forall( [remain[i,j] = rs[i,j] + rs[i+1,j] + cs[i,j] + cs[i,j+1] | i in 1..nrow, j in 1..ncol where remain[i,j] != -1]);

        % add continuous constraint
        % constraint for four corner
        constraint rs[1,1] = cs[1,1];
        constraint rs[1,ncol] = cs[1,ncol+1];
        constraint rs[nrow+1,1] = cs[nrow,1];
        constraint rs[nrow+1,ncol] = cs[nrow,ncol+1];
        % constraint for four margin
        constraint forall( [cs[i-1,1]+cs[i,1]=1 | i in 2..nrow where rs[i,1]=1]);
        constraint forall( [cs[i-1,ncol+1]+cs[i,ncol+1]=1 | i in 2..nrow where rs[i,ncol]=1]);
        constraint forall( [rs[1,j-1]+rs[1,j]=1 | j in 2..ncol where cs[1,j]=1]);
        constraint forall( [rs[nrow+1,j-1]+rs[nrow+1,j]=1 | j in 2..ncol where cs[nrow,j]=1]);

        constraint forall( [cs[i+1,1]+rs[i+1,1]=1 | i in 1..nrow-1 where cs[i,1]=1]);
        constraint forall( [cs[i-1,1]+rs[i,1]=1 | i in 2..nrow where cs[i,1]=1]);
        constraint forall( [cs[i+1,ncol+1]+rs[i+1,ncol]=1 | i in 1..nrow-1 where cs[i,ncol+1]=1]);
        constraint forall( [cs[i-1,ncol+1]+rs[i,ncol]=1 | i in 2..nrow where cs[i,ncol+1]=1]);

        constraint forall( [rs[1,j+1]+cs[1,j+1]=1 | j in 1..ncol-1 where rs[1,j]=1]);
        constraint forall( [rs[1,j-1]+cs[1,j]=1 | j in 2..ncol where rs[1,j]=1]);
        constraint forall( [rs[nrow+1,j+1]+cs[nrow,j+1]=1 | j in 1..ncol-1 where rs[nrow+1,j]=1]);
        constraint forall( [rs[nrow+1,j-1]+cs[nrow,j]=1 | j in 2..ncol where rs[nrow+1,j]=1]);
        % constraint for inner edges
        constraint forall( [rs[i,j-1] + cs[i,j] + cs[i-1,j] = 1 | i in 2..nrow, j in 2..ncol where rs[i,j] = 1]);
        constraint forall( [rs[i,j+1] + cs[i,j+1] + cs[i-1,j+1] = 1 | i in 2..nrow, j in 1..ncol-1 where rs[i,j] = 1]);
        constraint forall( [rs[i,j-1] + rs[i,j] + cs[i-1,j] = 1 | i in 2..nrow, j in 2..ncol where cs[i,j] = 1]);
        constraint forall( [rs[i+1,j-1] + rs[i+1,j] + cs[i+1,j] = 1 | i in 1..nrow-1, j in 2..ncol where cs[i,j] = 1]);
    """)

    row_forbid, col_forbid = find_forbid(problem.constraint)
    row_force, col_force = find_force(problem.constraint)

    for row_pos in row_forbid:
        model.add_string('constraint rs[' + str(row_pos[0]+1) + ',' + str(row_pos[1]+1) + '] = 0;')
    for row_pos in row_force:
        model.add_string('constraint rs[' + str(row_pos[0]+1) + ',' + str(row_pos[1]+1) + '] = 1;')
    for col_pos in col_forbid:
        model.add_string('constraint cs[' + str(col_pos[0]+1) + ',' + str(col_pos[1]+1) + '] = 0;')
    for col_pos in col_force:
        model.add_string('constraint cs[' + str(col_pos[0]+1) + ',' + str(col_pos[1]+1) + '] = 1;')

    solver = Solver.lookup(solver_name)
    instance = Instance(solver, model)

    # 传参数进去
    instance["nrow"] = problem.nrow
    instance["ncol"] = problem.ncol
    instance["remain"] = constraint_to_remain(problem.constraint)
    
    result = instance.solve()
    problem.row_solution = np.array(result["rs"])
    problem.col_solution = np.array(result["cs"])
    print()
    print("MINIZINC SOLUTION")
    problem.print_solution()


if __name__ == '__main__':
    # overall_limit = '222333'  #2*3
    # overall_limit = '323*221**3233213'
    # overall_limit = '*3****1**3*0203*2**3*22**'
    # overall_limit = '**32*22**2****1**0*2*333*'
    # nrow, ncol = 5, 5
    # problem = slitherlink(nrow, ncol, constraint=np.array(list(overall_limit)).reshape(nrow, ncol))
    problem = generate_problem_from_url(url='http://www.puzzle-loop.com/?v=0&size=0')
    problem.print_problem()

    start_time = time.time()
    minizinc_solve(problem, "gecode") # "chuffed", "gecode"
    end_time = time.time()
    print()
    print('start time: {}'.format(start_time))
    print('end time: {}'.format(end_time))
    print('time cost: {}'.format(end_time-start_time))
