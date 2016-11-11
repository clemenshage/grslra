function video_ialm
addpath('../../../matlab')
addpath('../../../matlab/LMaFit-SMS.v1/inexact_alm_rpca/PROPACK')
addpath('../../../matlab/LMaFit-SMS.v1/inexact_alm_rpca')

load('../escalator_130p')
X = double(X);
nofframes = size(X,2);

tau = 20;

mu = median(X,2);
X = bsxfun(@minus,X,mu);

tic
[L,~]=inexact_alm_rpca(X);
t = toc;

display(['Finished in ' num2str(t) ' seconds.'])

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

save('result_ialm.mat','L', 'X_masked','t','dimensions')

end
