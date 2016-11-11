function video_lmafit
addpath('../../../matlab')
addpath('../../../matlab/LMaFit-SMS.v1')

opts.tol = 1e-4;
opts.maxit = 200;
opts.est_rank = 0;
beta = [];


load('../escalator_130p')
X = double(X);
nofframes = size(X,2);
k = 4;

tau = 20;

mu = median(X,2);
X = bsxfun(@minus,X,mu);

tic
[A, B, ~, ~] = lmafit_sms_v1(X,k,opts,beta);
t = toc;

L = A*B;

display(['Finished in ' num2str(t) ' seconds.'])
display(['Final rank estimate: ' num2str(size(A,2))])

X = bsxfun(@plus,X,mu);
L = bsxfun(@plus,L,mu);

L(L < 0) = 0;
L(L > 255) = 255;

mask = (abs(X - L) > tau);
maskframes = reshape(mask, [dimensions(1) dimensions(2) nofframes]);

for i=1:nofframes
    maskframes(:,:,i) = medfilt2(maskframes(:,:,i), [3 3]);
end
mask = reshape(maskframes, size(X));
X_masked = X.*mask;
X_masked(mask == 0) = 255;
L = uint8(L);
X_masked = uint8(X_masked);

save('result_lmafit.mat','L', 'X_masked','t','dimensions')

end