function phase_transitions_lmafit
addpath('../../../matlab')
addpath('../../../matlab/LMaFit-SMS.v1')

% original parameters from demo_rand.m
% tol = 1e-4
% sigma = 0e-3; % noise STD
% tol = max(tol, .1*sigma); 
% opt.tol = min(1e-1,tol); 
% opt.maxit = 200;

% modified parameters for higher accuracy
opts.tol = 1e-12;
opts.maxit = 1000;
opts.est_rank = 0;
opts.print = 0;
beta = [];

m=200;
n=200;

maxval = 0.6;
step = 0.01;

values = round(maxval / step);
impossible_thresh = 2;

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
        [X, L_0, ~] = testdata_lmafit(m, n, k, rho, 0.01);
        tic
        [X, Y, ~, ~] = lmafit_sms_v1(X,k,opts,beta);
        t = toc;
        L = X*Y;
        err = norm(L_0 - L,'fro') / norm(L_0,'fro')
        result(it_k_m, it_rho, 1) = err;
        result(it_k_m, it_rho, 2) = t;
    end
    
end

save('result_lmafit.mat','result','maxval','step')
end