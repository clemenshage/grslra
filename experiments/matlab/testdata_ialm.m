function [X,L,S]=testdata_ialm(m,n,k,rho,amplitude)
U = 1/m * randn(m, k);
V = 1/n * randn(k, n);
L = U*V;
% U, R = qr(U,0);
% Y = R*V;

S = myspbernoulli(m, n, rho, amplitude);

X = L + S;

end

function S = myspbernoulli(m, n, rho, amplitude)
S = zeros(m, n);
nzentries = round(rho * m * n);
values = [-amplitude, amplitude];
entries = randsample(values, nzentries, true);
Omega = ind2sub([m, n], randsample(m * n, nzentries, false));
S(Omega) = entries;
end
