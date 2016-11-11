function completion_rmc
addpath('../../../matlab')
addpath(genpath('../../../matlab/RMC_1.1'))
dir = pwd;
cd('../../../matlab/manopt')
importmanopt
cd(dir)

rate_Omega=0.2;

m=200;
n=200;

card_Omega = round(rate_Omega*m*n);

maxval = 0.6;
step = 0.01;

values = round(maxval / step);
impossible_thresh = 1.2;

result = zeros(values, values, 2);
result(:,:,1)=1;

values = step:step:maxval;
nofvalues = numel(values);

% Original Parameters
% CG parameters
params.manopt.maxiter = 40 ;
params.manopt.verbosity = 0 ;
params.manopt.minstepsize = 0 ;
params.manopt.tolgradnorm = 1e-8 ;
% Outer loop parameters
params.huber.epsilon = 1 ;  % Good for synthetic experiments. For other problems, use the mean absolute value of the matrix M
params.huber.theta = 0.05 ; % Good for synthetic experiments. Can be increased (0.1, 1/2, etc.) for real datasets.
params.huber.tol = 1e-8 ; 
params.huber.itmax = 7 ;    % The maximum number of iteration
params.huber.verbose = 0 ;

load(['result_rmc_' num2str(100*rate_Omega) '.mat'],'result')

for it_k_m = 1:15
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
        ix = randsample(m * n, card_Omega, false);
        [Omega.I, Omega.J] = ind2sub([m, n], ix);
        X_Omega = X(ix);
        dimensions=[m n];
        
        [L,t]=rmc_wrapper(X_Omega, k, params, Omega, dimensions);
        err = norm(L_0 - L,'fro') / norm(L_0,'fro')
        result(it_k_m, it_rho, 1) = err;
        result(it_k_m, it_rho, 2) = t;
    end
    save(['result_rmc_' num2str(100*rate_Omega) '.mat'],'result','maxval','step')
end

end