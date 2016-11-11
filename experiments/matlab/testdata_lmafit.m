function [X,L,S]=testdata_lmafit(m,n,k,rho,sigma)
V = randn(m, k);
W = randn(k, n);
L = V * W;
% U, R = qr(U,0);
% Y = R*V;
S = mysprandn(m, n, rho, sigma * m);

X = L + S;

end

function S = mysprandn(m, n, rho, amplitude)
S = zeros(m, n);
nzentries = round(rho * m * n);
values = amplitude * randn(nzentries,1);
Omega = ind2sub([m, n], randsample(m * n, nzentries, false));
S(Omega) = values;
end