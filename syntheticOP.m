function [T, info] = syntheticOP(C, n, m, p, q, lambda, options)
 
%% Create the problem structure for the coupling matrix
% n \times m matrices with column rum to p and row sums to q.
% a space of matrices in dimension (n-1)(m-1).

problem.M = couplingmatrixfactory(n, m, p, q); 
problem.cost = @cost;
problem.egrad = @egrad;
%% Solve
[T, ~, info] = conjugategradient(problem,[], options); 

%% If we are changing a problem, the following is what to be changed
%% Cost function
% Here we use the entropy definition in M Cuturi (2013), NIPS
function [val] = cost(X)
    val = sum(X .* C + lambda*(X .* log(X)), 'all'); % the trace   
end

%% Euclidient Gradient
function [egradval] = egrad(X)
    egradval = C + lambda*( ones(size(X)) + log(X)); % change to ones(size (X) if necessary 
end
end
