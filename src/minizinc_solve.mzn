% Slitherlink
int: nrow = 3;
int: ncol = 3;
set of int: digit_nrow_range = 1..nrow;
set of int: digit_ncol_range = 1..ncol;
set of int: rs_row_range = 1..nrow+1;
set of int: cs_col_range = 1..ncol+1;
array[1..nrow+1,1..ncol] of var 0..1: rs;
array[1..nrow,1..ncol+1] of var 0..1: cs;
array[1..nrow,1..ncol] of int: remain;
remain = [| 3,2,1,| 2,1,2,| 2,2,2|]; % use -1 to represent null constraint

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
constraint forall( [rs[nrow,j-1]+rs[nrow,j]=1 | j in 2..ncol where cs[nrow,j]=1]);
% constraint for inner edges
constraint forall( [rs[i,j-1] + cs[i,j] + cs[i-1,j] = 1 | i in 2..nrow, j in 2..ncol-1 where rs[i,j] = 1]);
constraint forall( [rs[i,j+1] + cs[i,j+1] + cs[i-1,j+1] = 1 | i in 2..nrow, j in 2..ncol-1 where rs[i,j] = 1]);
constraint forall( [rs[i,j-1] + rs[i,j] + cs[i-1,j] = 1 | i in 2..nrow-1, j in 2..ncol where cs[i,j] = 1]);
constraint forall( [rs[i+1,j-1] + rs[i+1,j] + cs[i+1,j] = 1 | i in 2..nrow-1, j in 2..ncol where cs[i,j] = 1]);

solve satisfy;

output ["rs=\(rs)\ncs=\(cs)\n"];
