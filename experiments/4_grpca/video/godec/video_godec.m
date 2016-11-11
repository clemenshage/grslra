function video_godec
addpath('../../../matlab')
addpath('../../../matlab/GoDec')

load('../escalator_130p')
X = double(X);
nofframes = size(X,2);

tau = 20;

mu = median(X,2);
X = bsxfun(@minus,X,mu);

tic
[L,S]=GoDec(X, 4, 0.03*numel(X),0);
t = toc;

display(['Finished in ' num2str(t) ' seconds.'])

X = bsxfun(@plus,X,mu);
L = bsxfun(@plus,L,mu);

L(L < 0) = 0;
L(L > 255) = 255;

mask = (abs(S) > tau);
maskframes = reshape(mask, [dimensions(1) dimensions(2) nofframes]);

for i=1:nofframes
    maskframes(:,:,i) = medfilt2(maskframes(:,:,i), [3 3]);
end
mask = reshape(maskframes, size(X));
X_masked = X.*mask;
X_masked(mask == 0) = 255;
X_masked = uint8(X_masked);
L = uint8(L);

save('result_godec.mat','L', 'X_masked','t','dimensions')

end