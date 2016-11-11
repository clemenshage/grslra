function [A_rec,D, t]=grasta_wrapper(A,k)
%% First we generate the data matrix and incomplete sample vector.
grasta_path; % Add search path

% s = RandStream('swb2712','Seed',1600);
% RandStream.setDefaultStream(s);

SAMPLING   = 1;
%noiseFac   = 1 * 1e-5;

% Number of rows and columns
[numr numc] =size(A);

% Rank of the underlying matrix.
truerank = k;

% Size of vectorized matrix
N = numr*numc;
% Number of samples that we will reveal.
M = round(SAMPLING * N);

% Select a random set of M entries of Y.
p = randperm(N);
idx = p(1:M);
clear p;

[I,J] = ind2sub([numr,numc],idx(1:M));
[J, inxs]=sort(J'); I=I(inxs)';

I=repmat(1:numr,1,numc);
S=reshape(A,numel(A),1);


% %% Now we set parameters for the algorithm.
% % We set the number of cycles and put the necessary parameters into OPTIONS
% 
% maxCycles                   = 20;    % the max cycles of robust mc
% OPTIONS.QUIET               = 1;     % suppress the debug information
% OPTIONS.ABSTOL_THRESHOLD    = 1e-6;
% OPTIONS.RELTOL_THRESHOLD    = 1e-3;
% OPTIONS.ITER_MIN            = 10;    % the min iteration allowed for ADMM in the beginning
% OPTIONS.ITER_MAX            = 100;   % the max iteration allowed for ADMM
% OPTIONS.DIM_M               = numr;  % your data's ambient dimension
% OPTIONS.RANK                = truerank; % give your estimated rank
% OPTIONS.rho                 = 1.8;   % constant ADMM step-size
% OPTIONS.USE_MEX             = 0;     % If you do not have the mex-version of Alg 2
%                                      % please set Use_mex = 0.
% 
% CONVERGE_LEVLE              = 16;    % when status.level >= CONVERGE_LEVLE, we find the subspace

% Now we set parameters for the algorithm.
% We set the number of cycles and put the necessary parameters into OPTIONS

maxCycles                   = 10;    % the max cycles of robust mc
OPTIONS.QUIET               = 1;     % suppress the debug information

OPTIONS.MAX_LEVEL           = 20;    % For multi-level step-size,
OPTIONS.MAX_MU              = 15;    % For multi-level step-size
OPTIONS.MIN_MU              = 1;     % For multi-level step-size

OPTIONS.DIM_M               = numr;  % your data's ambient dimension
OPTIONS.RANK                = truerank; % give your estimated rank

OPTIONS.ITER_MIN            = 20;    % the min iteration allowed for ADMM at the beginning
OPTIONS.ITER_MAX            = 20;    % the max iteration allowed for ADMM
OPTIONS.rho                 = 2;   % ADMM penalty parameter for acclerated convergence
OPTIONS.TOL                 = 1e-8;   % ADMM convergence tolerance

OPTIONS.USE_MEX             = 0;     % If you do not have the mex-version of Alg 2
                                     % please set Use_mex = 0.
                                     
CONVERGE_LEVLE              = 20;    % If status.level >= CONVERGE_LEVLE, robust mc converges

%% % % % % % % % % % % % % % % % % % % % %
% Now run robust matrix completion.
tic;
[Usg, Vsg, Osg] = grasta_mc(I,J,S,numr,numc,maxCycles,CONVERGE_LEVLE,OPTIONS);
t = toc;
A_rec=Usg*Vsg';
D = Osg';

end
