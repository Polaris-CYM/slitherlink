% Slitherlink
int: nrow = 3;
int: ncol = 3;
set of int: digit_nrow_range = 1..nrow;
set of int: digit_ncol_range = 1..ncol;
% set of int: rs_row_range = 1..nrow+1;
% set of int: cs_col_range = 1..ncol+1;
array[1..nrow+1,1..ncol] of var 0..1: rs; % rs是一个形状为nrow+1 x ncol+1的二维数组，其中每个变量的值域为0到1（也就是0或1）
array[1..nrow,1..ncol+1] of var 0..1: cs;
array[1..nrow,1..ncol] of int: remain;
remain = [| 3,2,1,| 2,1,2,| 2,2,2|]; % use -1 to represent null constraint

% add digital constraint
% constraint forall(i in digit_nrow_range, j in digit_ncol_range)(remain[i,j] = rs[i,j] + rs[i+1,j] + cs[i,j] + cs[i,j+1]);
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
constraint forall( [rs[i,j-1] + cs[i,j] + cs[i-1,j] = 1 | i in 2..nrow, j in 2..ncol-1 where rs[i,j] = 1]);
constraint forall( [rs[i,j+1] + cs[i,j+1] + cs[i-1,j+1] = 1 | i in 2..nrow, j in 2..ncol-1 where rs[i,j] = 1]);
constraint forall( [rs[i,j-1] + rs[i,j] + cs[i-1,j] = 1 | i in 2..nrow-1, j in 2..ncol where cs[i,j] = 1]);
constraint forall( [rs[i+1,j-1] + rs[i+1,j] + cs[i+1,j] = 1 | i in 2..nrow-1, j in 2..ncol where cs[i,j] = 1]);

% Edges that need to be forbidden when two 3's are connected
constraint forall( [cs[i-1,j+1] + cs[i+1,j+1] = 0 | i in 2..nrow-1, j in 1..ncol-1 where remain[i,j] = 3 /\ remain[i,j+1] = 3]);
constraint forall( [rs[i+1,j-1] + rs[i+1,j+1] = 0 | i in 1..nrow-1, j in 2..ncol-1 where remain[i,j] = 3 /\ remain[i+1,j] = 3]);
constraint forall( [cs[2,j+1] = 0 | j in 1..ncol-1 where remain[1,j] = 3 /\ remain[1,j+1] = 3]);
constraint forall( [cs[nrow-1,j+1] = 0 | j in 1..ncol-1 where remain[nrow,j] = 3 /\ remain[nrow,j+1] = 3]);
constraint forall( [rs[i+1,2] = 0 | i in 1..nrow-1 where remain[i,1] = 3 /\ remain[i+1,1] = 3]);
constraint forall( [rs[i+1,ncol-1] = 0 | i in 1..nrow-1 where remain[i,ncol] = 3 /\ remain[i+1,ncol] = 3]);
        
% Edges that need to be forbidden when 1 is in the corner
constraint if remain[1,1] = 1 then rs[1,1] + cs[1,1] = 0  endif;
constraint if remain[1,ncol] = 1 then rs[1,ncol] + cs[1,ncol+1] = 0  endif;
constraint if remain[nrow,1] = 1 then rs[nrow+1,1] + cs[nrow,1] = 0  endif;
constraint if remain[nrow,ncol] = 1 then rs[nrow+1,ncol] + cs[nrow,ncol+1] = 0  endif;
        
% Edges that must be chosen when 3 is in the corner
constraint if remain[1,1] = 3 then rs[1,1] + cs[1,1] = 2  endif;
constraint if remain[1,ncol] = 3 then rs[1,ncol] + cs[1,ncol+1] = 2  endif;
constraint if remain[nrow,1] = 3 then rs[nrow+1,1] + cs[nrow,1] = 2  endif;
constraint if remain[nrow,ncol] = 3 then rs[nrow+1,ncol] + cs[nrow,ncol+1] = 2  endif;
        
% Edges that must be chosen when 2 is in the corner
constraint if remain[1,1] = 2 then rs[1,2] + cs[2,1] = 2  endif;
constraint if remain[1,ncol] = 2 then rs[1,ncol-1] + cs[2,ncol+1] = 2  endif;
constraint if remain[nrow,1] = 2 then rs[nrow+1,2] + cs[nrow-1,1] = 2  endif;
constraint if remain[nrow,ncol] = 2 then rs[nrow+1,ncol-1] + cs[nrow-1,ncol+1] = 2  endif;

solve satisfy;

output ["rs=\(rs)\ncs=\(cs)\n"];
