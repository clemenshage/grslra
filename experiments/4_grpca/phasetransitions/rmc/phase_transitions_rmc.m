function phase_transitions_rmc
addpath('../../../matlab')
addpath(genpath('../../../matlab/RMC_1.1'))
dir = pwd;
cd('../../../matlab/manopt')
importmanopt
cd(dir)

% High accuracy parameters
% CG parameters
params.manopt.maxiter = 40 ;
params.manopt.minstepsize = 0 ;
params.manopt.tolgradnorm = 1e-8 ;
% Outer loop parameters
params.huber.epsilon = 1 ;  % Good for synthetic experiments. For other problems, use the mean absolute value of the matrix M
params.huber.theta = 0.5 ; % Good for synthetic experiments. Can be increased (0.1, 1/2, etc.) for real datasets.
params.huber.tol = 1e-16 ; 
params.huber.itmax = 1000 ;    % The maximum number of iteration
params.huber.verbose = 0 ;
params.manopt.verbosity = 0 ;

m=200;
n=200;

maxval = 0.6;
step = 0.01;

values = round(maxval / step);
impossible_thresh = 1.2;

result = zeros(values, values, 2);


values = step:step:maxval;
nofvalues = numel(values);

for it_k_m = 1:nofvalues
    k_m = values(it_k_m);
    k = round(k_m * m);
    for it_rho = 1:nofvalues
        rho = values(it_rho);
        display(['relative rank: ', num2str(k_m), ' rho: ', num2str(rho)])
        if it_k_m + it_rho > impossible_thresh * nofvalues
            result(it_k_m, it_rho, 1) = 1;
            result(it_k_m, it_rho, 2) = 0;
            continue
        end
        [X, L_0, S_0] = testdata_lmafit(m, n, k, rho, 0.01);
        [L,t]=rmc_wrapper(X, k, params);
        err = norm(L_0 - L,'fro') / norm(L_0,'fro')
        result(it_k_m, it_rho, 1) = err;
        result(it_k_m, it_rho, 2) = t;
    end
    save('result_rmc.mat','result','maxval','step')
end

end