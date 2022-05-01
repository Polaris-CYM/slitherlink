import numpy as np

'''(X, D, C)
x: variables, self.row_solution + self.col_solution
d: Domain, {0, 1}, {True, False}
c: constrain, the relationship between x
            1. constraint[i,j] = row_solution[i,j] + row_solution[i+1,j] + col_solution[i,j] + col_solution[i,j+1]
              digital limit             up                    down                left                  right
            2. closed circle
'''


class slitherlink:
    def __init__(self, nrow, ncol, constraint):
        """
        :param nrow: length of the rows of the puzzle
        :param ncol: length of the columns of the puzzle
        :param constraint: a 2-dimention numpy array, with shape=(nrow, ncol)
        """
        self.nrow = nrow
        self.ncol = ncol
        self.constraint = constraint  # All digit constraints: 0, 1, 2, 3, and *(no limit)

        # self.row_solution[i,j] (i <= nrow) : the upper edge of constraint[i,j]（i < nrow）
        # self.col_solution[i,j] : the edge to the left of constraint[i,j]
        # row_solution[i,j] = 1 if the solution includes this horizontal edge
        self.row_solution = np.zeros(shape=(self.nrow+1, self.ncol))  # (r+1) * c
        self.col_solution = np.zeros(shape=(self.nrow, self.ncol+1))  # r * (c+1)

    def print_problem(self):
        """
        Print the origin puzzle (without solution) on the screen
        """
        print("-" * (2 * self.ncol + 1))
        for line in self.constraint:
            print('|' + '|'.join(map(str, line)) + '|')
            print("-" * (2 * self.ncol + 1))

    def print_solution(self):
        """
        Print the solution as well as the origin puzzle on the screen
        """
        for i in range(self.nrow):
            up_str = ' '
            for j in self.row_solution[i]:
                up_str += '-' if j == 1 else ' '
                up_str += ' '
            print(up_str)

            line_str = ''
            for j in range(self.ncol):
                line_str += '|' if self.col_solution[i,j] == 1 else ' '
                line_str += self.constraint[i,j]
            line_str += '|' if self.col_solution[i,self.ncol] == 1 else ' '
            print(line_str)
        
        down_str = ' '
        for j in self.row_solution[self.nrow]:
            down_str += '-' if j == 1 else ' '
            down_str += ' '
        print(down_str)


if __name__ == '__main__':
    """
    nrow = 20
    ncol = 15
    limit = '**2***02*21*12***3**2**30***2**3***33**3*3***1**1**1**1***3**0***1***30***3***3*'
    limit += '**32**13****01**31**1***2***3**1***2***22***3***3**3***2***0*30**10**11**31**23*'
    limit += '2***3***3**3***1***12***2***1**3***1***2**03**13****32**01***3***3***13***3***0*'
    limit += '*1***3**3**1**3***3*3**23***2**2***03**0**3***13*22*20***0**'
    """
    nrow = 4
    ncol = 4
    limit = '1223*1133113322*'

    limit = np.array(list(limit)).reshape(nrow, ncol)

    problem = slitherlink(nrow, ncol, limit)
    problem.print_problem()

    problem.row_solution[0,2] = problem.row_solution[0,3] = 1
    problem.row_solution[1,0] = problem.row_solution[1,1] = problem.row_solution[1,3] = 1
    problem.row_solution[2,0] = problem.row_solution[2,3] = 1
    problem.row_solution[3,0] = problem.row_solution[3,2] = problem.row_solution[3,3] = 1
    problem.row_solution[4,0] = problem.row_solution[4,1] = 1
    problem.col_solution[0,2] = problem.col_solution[0,4] = 1
    problem.col_solution[1,0] = problem.col_solution[1,3] = 1
    problem.col_solution[2,1] = problem.col_solution[2,4] = 1
    problem.col_solution[3,0] = problem.col_solution[3,2] = 1
    problem.print_solution()
