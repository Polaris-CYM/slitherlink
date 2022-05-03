import time
from generate_problem import *
from constraint_propagation_forward import *
from minizinc import Instance, Model, Solver


def constraint_to_remain(constraint):
    # "constraint" is a two-dimensional array of strings, which needs to be converted into a list
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

    # All puzzles must add these constraints, and the nrow and ncol inside will be passed parameters later.
    model.add_string("""
        int: nrow;
        int: ncol;
        array[1..nrow+1,1..ncol] of var 0..1: rs;
        array[1..nrow,1..ncol+1] of var 0..1: cs;
        array[1..nrow,1..ncol] of int: remain;

        % add digital constraint
        constraint forall( [remain[i,j] = rs[i,j] + rs[i+1,j] + cs[i,j] + cs[i,j+1] | i in 1..nrow, j in 1..ncol where remain[i,j] != -1]);

        % add constraints on continuity
        % constraint on four corners
        constraint rs[1,1] = cs[1,1];
        constraint rs[1,ncol] = cs[1,ncol+1];
        constraint rs[nrow+1,1] = cs[nrow,1];
        constraint rs[nrow+1,ncol] = cs[nrow,ncol+1];
        
        % constraints on edges intersecting boundaries
        constraint forall( [cs[i-1,1]+cs[i,1]=1 | i in 2..nrow where rs[i,1]=1]);
        constraint forall( [cs[i-1,ncol+1]+cs[i,ncol+1]=1 | i in 2..nrow where rs[i,ncol]=1]);
        constraint forall( [rs[1,j-1]+rs[1,j]=1 | j in 2..ncol where cs[1,j]=1]);
        constraint forall( [rs[nrow+1,j-1]+rs[nrow+1,j]=1 | j in 2..ncol where cs[nrow,j]=1]);
        
        % constraint on four boundaries
        constraint forall( [cs[i+1,1]+rs[i+1,1]=1 | i in 1..nrow-1 where cs[i,1]=1]);
        constraint forall( [cs[i-1,1]+rs[i,1]=1 | i in 2..nrow where cs[i,1]=1]);
        constraint forall( [cs[i+1,ncol+1]+rs[i+1,ncol]=1 | i in 1..nrow-1 where cs[i,ncol+1]=1]);
        constraint forall( [cs[i-1,ncol+1]+rs[i,ncol]=1 | i in 2..nrow where cs[i,ncol+1]=1]);

        constraint forall( [rs[1,j+1]+cs[1,j+1]=1 | j in 1..ncol-1 where rs[1,j]=1]);
        constraint forall( [rs[1,j-1]+cs[1,j]=1 | j in 2..ncol where rs[1,j]=1]);
        constraint forall( [rs[nrow+1,j+1]+cs[nrow,j+1]=1 | j in 1..ncol-1 where rs[nrow+1,j]=1]);
        constraint forall( [rs[nrow+1,j-1]+cs[nrow,j]=1 | j in 2..ncol where rs[nrow+1,j]=1]);
        
        % constraint on inner edges
        constraint forall( [rs[i,j-1] + cs[i,j] + cs[i-1,j] = 1 | i in 2..nrow, j in 2..ncol where rs[i,j] = 1]);
        constraint forall( [rs[i,j+1] + cs[i,j+1] + cs[i-1,j+1] = 1 | i in 2..nrow, j in 1..ncol-1 where rs[i,j] = 1]);
        constraint forall( [rs[i,j-1] + rs[i,j] + cs[i-1,j] = 1 | i in 2..nrow, j in 2..ncol where cs[i,j] = 1]);
        constraint forall( [rs[i+1,j-1] + rs[i+1,j] + cs[i+1,j] = 1 | i in 1..nrow-1, j in 2..ncol where cs[i,j] = 1]);
        
        % Edges that need to be forbidden when two 3's are connected
        constraint forall( [cs[i-1,j+1] + cs[i+1,j+1] = 0 | i in 2..nrow-1, j in 1..ncol-1 where remain[i,j] = 3 /\ remain[i,j+1] = 3]);
        constraint forall( [rs[i+1,j-1] + rs[i+1,j+1] = 0 | i in 1..nrow-1, j in 2..ncol-1 where remain[i,j] = 3 /\ remain[i+1,j] = 3]);
        constraint forall( [cs[2,j+1] = 0 | j in 1..ncol-1 where remain[1,j] = 3 /\ remain[1,j+1] = 3]);
        constraint forall( [cs[nrow-1,j+1] = 0 | j in 1..ncol-1 where remain[nrow,j] = 3 /\ remain[nrow,j+1] = 3]);
        constraint forall( [rs[i+1,2] = 0 | i in 1..nrow-1 where remain[i,1] = 3 /\ remain[i+1,1] = 3]);
        constraint forall( [rs[i+1,ncol-1] = 0 | i in 1..nrow-1 where remain[i,ncol] = 3 /\ remain[i+1,ncol] = 3]);
        
        % Edges that need to be forbidden when a '1' is in the corner
        constraint if remain[1,1] = 1 then rs[1,1] + cs[1,1] = 0  endif;
        constraint if remain[1,ncol] = 1 then rs[1,ncol] + cs[1,ncol+1] = 0  endif;
        constraint if remain[nrow,1] = 1 then rs[nrow+1,1] + cs[nrow,1] = 0  endif;
        constraint if remain[nrow,ncol] = 1 then rs[nrow+1,ncol] + cs[nrow,ncol+1] = 0  endif;
        
        % Edges that must be chosen when a '3' is in the corner
        constraint if remain[1,1] = 3 then rs[1,1] + cs[1,1] = 2  endif;
        constraint if remain[1,ncol] = 3 then rs[1,ncol] + cs[1,ncol+1] = 2  endif;
        constraint if remain[nrow,1] = 3 then rs[nrow+1,1] + cs[nrow,1] = 2  endif;
        constraint if remain[nrow,ncol] = 3 then rs[nrow+1,ncol] + cs[nrow,ncol+1] = 2  endif;
        
        % Edges that must be chosen when a '2' is in the corner
        constraint if remain[1,1] = 2 then rs[1,2] + cs[2,1] = 2  endif;
        constraint if remain[1,ncol] = 2 then rs[1,ncol-1] + cs[2,ncol+1] = 2  endif;
        constraint if remain[nrow,1] = 2 then rs[nrow+1,2] + cs[nrow-1,1] = 2  endif;
        constraint if remain[nrow,ncol] = 2 then rs[nrow+1,ncol-1] + cs[nrow-1,ncol+1] = 2  endif;

        % Edges that must be chosen when '0' and '3' are connected
        constraint forall( [rs[i,j-1] + rs[i+1,j] + rs[i,j+1] + cs[i,j] + cs[i,j+1] = 5 | i in 2..nrow, j in 2..ncol-1 where remain[i,j] = 3 /\ remain[i-1,j] = 0]);
        constraint forall( [rs[i+1,j-1] + rs[i,j] + rs[i+1,j+1] + cs[i,j] + cs[i,j+1] = 5 | i in 1..nrow-1, j in 2..ncol-1 where remain[i,j] = 3 /\ remain[i+1,j] = 0]);
        constraint forall( [cs[i-1,j] + cs[i,j+1] + cs[i+1,j] + rs[i,j] + rs[i+1,j] = 5 | i in 2..nrow-1, j in 2..ncol where remain[i,j] = 3 /\ remain[i,j-1] = 0]);
        constraint forall( [cs[i-1,j+1] + cs[i,j] + cs[i+1,j+1] + rs[i,j] + rs[i+1,j] = 5 | i in 2..nrow-1, j in 1..ncol-1 where remain[i,j] = 3 /\ remain[i,j+1] = 0]);
        
        % Edges that must be chosen when a '3' is adjacent to a '0' diagonally
        constraint forall( [cs[i,j+1] + rs[i+1,j] = 2 | i in 1..nrow-1, j in 1..ncol-1 where remain[i,j] = 3 /\ remain[i+1,j+1] = 0]);
        constraint forall( [cs[i,j] + rs[i+1,j] = 2 | i in 1..nrow-1, j in 2..ncol where remain[i,j] = 3 /\ remain[i+1,j-1] = 0]);
        constraint forall( [cs[i,j] + rs[i,j] = 2 | i in 2..nrow, j in 2..ncol where remain[i,j] = 3 /\ remain[i-1,j-1] = 0]);
        constraint forall( [cs[i,j+1] + rs[i,j] = 2 | i in 2..nrow, j in 1..ncol-1 where remain[i,j] = 3 /\ remain[i-1,j+1] = 0]);
        
        % Edges that must be chosen when a '3' is adjacent to a '3' diagonally
        constraint forall( [cs[i,j] + rs[i,j] + cs[i+1,j+2] + rs[i+2,j+1] = 4 | i in 1..nrow-1, j in 1..ncol-1 where remain[i,j] = 3 /\ remain[i+1,j+1] = 3]);
        constraint forall( [cs[i,j+1] + rs[i,j] + cs[i+1,j-1] + rs[i+2,j-1] = 4 | i in 1..nrow-1, j in 2..ncol where remain[i,j] = 3 /\ remain[i+1,j-1] = 3]);
        constraint forall( [cs[i,j+1] + rs[i+1,j] + cs[i-1,j-1] + rs[i-1,j-1] = 4  | i in 2..nrow, j in 2..ncol where remain[i,j] = 3 /\ remain[i-1,j-1] = 3]);
        constraint forall( [cs[i,j] + rs[i+1,j] + cs[i-1,j+2] + rs[i-1,j+1] = 4  | i in 2..nrow, j in 1..ncol-1 where remain[i,j] = 3 /\ remain[i-1,j+1] = 3]);
        
        % Edges that mush be chosen when two 3s are in the same diagonal, but separated by a '2'
        %constraint forall( [cs[i,j] + rs[i,j] + cs[i+2,j+3] + rs[i+3,j+2] = 4 | i in 1..nrow-2, j in 1..ncol-2 where remain[i,j] = 3 /\ remain[i+1,j+1] = 2 /\ remain[i+2,j+2] = 3]);
        %constraint forall( [cs[i,j+1] + rs[i,j] + cs[i+2,j-2] + rs[i+3,j-2] = 4 | i in 1..nrow-2, j in 3..ncol where remain[i,j] = 3 /\ remain[i+1,j-1] = 2 /\ remain[i+2,j-2] = 3]);
        %constraint forall( [cs[i,j+1] + rs[i+1,j] + cs[i-2,j-2] + rs[i-2,j-2] = 4  | i in 3..nrow, j in 3..ncol where remain[i,j] = 3 /\ remain[i-1,j-1] = 2 /\ remain[i-2,j-2] = 3]);
        %constraint forall( [cs[i,j] + rs[i+1,j] + cs[i-2,j+3] + rs[i-2,j+2] = 4  | i in 3..nrow, j in 1..ncol-2 where remain[i,j] = 3 /\ remain[i-1,j+1] = 2 /\ remain[i-2,j+2] = 3]);

    """)

    # row_forbid, col_forbid = find_forbid(problem.constraint)
    # row_force, col_force = find_force(problem.constraint)
    #
    # for row_pos in row_forbid:
    #     model.add_string('constraint rs[' + str(row_pos[0]+1) + ',' + str(row_pos[1]+1) + '] = 0;')
    # for row_pos in row_force:
    #     model.add_string('constraint rs[' + str(row_pos[0]+1) + ',' + str(row_pos[1]+1) + '] = 1;')
    # for col_pos in col_forbid:
    #     model.add_string('constraint cs[' + str(col_pos[0]+1) + ',' + str(col_pos[1]+1) + '] = 0;')
    # for col_pos in col_force:
    #     model.add_string('constraint cs[' + str(col_pos[0]+1) + ',' + str(col_pos[1]+1) + '] = 1;')

    solver = Solver.lookup(solver_name)
    instance = Instance(solver, model)

    # pass parameters
    instance["nrow"] = problem.nrow
    instance["ncol"] = problem.ncol
    instance["remain"] = constraint_to_remain(problem.constraint)
    
    result = instance.solve()
    problem.row_solution = np.array(result["rs"])
    problem.col_solution = np.array(result["cs"])
    print()
    print("MINIZINC SOLUTION:")
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
