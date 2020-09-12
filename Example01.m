%please install manopt before run the code thank you 
clear all
seeds = 1;
rng(seeds*1000,'twister');
%addpath('../CM_Programs/')
%%addpath('/Users/daishi/Dropbox/DaiShiProjects/codes')
%%

C = [2, 2, 0.8, 0, 0;
    0, -2, -2, -2, 2;
    1, 2, 2, 2, -1;
    2, 1, 0, 1, -1;
    1, 1, 2, 1, 0;
    0, 1, 0, 1.2, -1;
    -2, 2, 2, 1, 1;
2, 1, 2, 1, -1]; 

C = -(C - 2);    % preferrence to cost

p = [3,3,3,4,2,2,2,1]'; 
q = [4 2 6 4 4]';
n = 8;
m = 5;

% Test One:  The classic OT
options.checkperiod = 10;
options.maxiter = 100;
options.verbosity = 2;
options.minstepsize = 1e-20;
lambda = 0.0;

[T1, info] = syntheticOP(C, n, m, p, q, lambda,options);
%%
 

% Prepare Linear Programming
options1 = optimoptions('linprog','Algorithm','interior-point');
f = C(:);
beq = [p; q];
Aeq = [kron(ones(1,m), eye(n)); kron(eye(m), ones(1,n))];
%x = linprog(f,[],[],Aeq,beq,zeros(n*m,1),[],options);
x = linprog(f,[],[],Aeq,beq,0.0*ones(n*m,1),[]);

T2 = reshape(x, n,m);
figure
subplot(221) 
imagesc(T1)
figure
subplot(221) 
imagesc(T2)

norm(T1-T2,'fro')/norm(T1,'fro')
%%

% Test Two:  The Entropy Regularized OT
options.checkperiod = 10;
options.maxiter = 100;
options.verbosity = 1;
options.minstepsize = 1e-20;
%lambda = 0.05;
% Options for Sinkhorn
p_norm = inf;
tolerance=.5e-2;
maxIter=100;
VERBOSE = 1;
   
lambda = logspace(-3,2,100);
difference = zeros(numel(lambda),1);
for ii = 1:numel(lambda)
   % CM algoritm
   tic
   [T3, info] = syntheticOP(C, n, m, p, q, lambda(ii), options);
   TCMM(ii) =toc;
   % The Sinkhorn algorithm
   newlambda = 1/lambda(ii);   % This is becaue sinkhormTransport.m use 1/lambda as regularizer
   K=exp(-newlambda*C);
   U=K.*C;
   % Call the dependency "sinkhornTransport.m" to solve the matrix scaling
   % problem
   tic
   [dis,lowerEMD,ll,mm]=sinkhornTransport(p,q,K,U,newlambda,[],p_norm,tolerance,maxIter,VERBOSE);
   T4=bsxfun(@times,mm',(bsxfun(@times,ll,K)));% this is the optimal transport
   Tsinkhorn(ii) =toc;
%    figure
%    subplot(221) 
%    imagesc(T3)
%    figure
%    subplot(221) 
%    imagesc(T4)
   difference(ii) = mean((T3-T4).^2, 'all');%norm(T3-T4,'fro')/norm(T4,'fro');   %();        %
   T_difference(ii)=TCMM(ii)-Tsinkhorn(ii)
end
%%
data =[reshape(lambda,100,1),reshape(T_difference,100,1),difference]
figure
subplot(221)
plot(difference)
xlabel('10^{-3} < \lambda  < 10^3')
ylabel('mean squared error ')
xlim([0.001,100])

%%
figure
subplot(221)
plot(lambda, T_difference)
xlabel('10^{-3} < \lambda  < 10^3')
xlim([0.001,100])
ylabel('Time difference (CMM-Sinkhorn)')
xlim([0.001,100])



