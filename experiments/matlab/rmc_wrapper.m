function [L, time] = rmc_wrapper(X, k, params, Omega, dimensions)

problem.r = k;

switch(nargin)
    case 3
        [problem.m, problem.n] = size(X);
        [problem.I, problem.J] = ind2sub(size(X), 1:(problem.m * problem.n));
        problem.I = problem.I';
        problem.J = problem.J';
    case 5
        problem.m = dimensions(1);
        problem.n = dimensions(2);
        problem.I = Omega.I;
        problem.J = Omega.J;
    otherwise
        display('error')
end

problem.Iu = uint32(problem.I);
problem.Ju = uint32(problem.J);
problem.A = reshape(X, [numel(X), 1]);
problem.lambda = 0; 
problem.k = numel(X);
problem.Atrue = zeros(size(problem.A));
problem.Utrue = zeros(problem.m, problem.r);
problem.Vtrue = zeros(problem.r, problem.n);

tic

[U, V] = rmc(problem, params);

time = toc;
L = U*V;

end