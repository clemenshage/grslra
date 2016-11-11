function phase_transitions_godec
addpath('../../../matlab')
addpath('../../../matlab/GoDec')

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
        tic
        [L,~]=GoDec(X, k, nnz(S_0),0);
        t = toc;
        err = norm(L_0 - L,'fro') / norm(L_0,'fro')
        result(it_k_m, it_rho, 1) = err;
        result(it_k_m, it_rho, 2) = t;
    end
    save('result_godec.mat','result','maxval','step')
end

end