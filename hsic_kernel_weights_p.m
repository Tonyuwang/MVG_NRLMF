function [weight_v] = hsic_kernel_weights_p(Kernels_list,adjmat,dim,regcoef1,regcoef2)
%regcoef1=0.01
%regcoef2=0.001
%
%

num_kernels = size(Kernels_list,3);%返回Kernels_list的维度，本实验中是4维

weight_v = zeros(num_kernels,1);%生成num_kernels行一列的零矩阵，本实验中是4行一列

y = adjmat;
    % Graph based kernel
if dim == 1
        ideal_kernel = y*y';%形成N*N阶的矩阵
else
        ideal_kernel = y'*y;
end
%ideal_kernel=Knormalized(ideal_kernel);

N_U = size(ideal_kernel,1);%返回ideal_kernel矩阵的行数
l=ones(N_U,1);
H = eye(N_U) - (l*l')/N_U;%eye(N_U)生成与ideal_kernel矩阵行数*行数的单位矩阵
%eye(N_U)文献中的I，H相当于文献中H，N_U相当于N
M = zeros(num_kernels,num_kernels);%生成4*4的零矩阵

for i=1:num_kernels %计算核矩阵之间的相似性
	for j=1:num_kernels
		kk1 = H*Kernels_list(:,:,i)*H;
		kk2 = H*Kernels_list(:,:,j)*H;
		%kk1 = Kernels_list(:,:,i);
		%kk2 = Kernels_list(:,:,j);
		
		%mm = dot( a1,a2 )/( sqrt( sum( a1.*a1 ) ) * sqrt( sum( a2.*a2 ) ) );
		mm = trace(kk1'*kk2);
		m1 = trace(kk1*kk1');
		m2 = trace(kk2*kk2');
		M(i,j) = mm/(sqrt(m1)*sqrt(m2));%文献中的W
		%M(i,j) = (mm);
	end
end
%d_1 = sum(M);
%D_1 = diag(d_1);
%LapM = D_1 - M;
%d_tmep_1=eye(num_kernels)/(D_1^(1/2));
%LapM = d_tmep_1*L_D_1*d_tmep_1;


a = zeros(num_kernels,1);%生成num_kernels行一列的零矩阵

for i=1:num_kernels%计算文献中tr那一项

	kk = H*Kernels_list(:,:,i)*H;
	aa = trace(kk'*ideal_kernel);
	a(i) = aa*(N_U-1)^-2;
end

v = randn(num_kernels,1);%随机生成4行一列的矩阵
falpha = @(v)obj_function(v,M,a,regcoef1,regcoef2);
[x_alpha, fval_alpha] = optimize_weights(v, falpha);
%fval_alpha
%weight_v = x_alpha/sum(x_alpha);
weight_v = x_alpha;
end

function [J] = obj_function(w,Mi,ai,regcoef1,regcoef2)
    
    J =  -regcoef1*(w'*ai) + regcoef2*w'*Mi*w ;
end

function [x, fval] = optimize_weights(x0, fun)
    n = length(x0);
    Aineq   = [];
    bineq   = [];
    Aeq     = ones(1,n);
    beq     = 1;
    LB      = zeros(1,n);
    UB      = ones(1,n);

    options = optimoptions('fmincon','Algorithm','interior-point', 'Display', 'notify');
    [x,fval] = fmincon(fun,x0,Aineq,bineq,Aeq,beq,LB,UB,[],options);
end