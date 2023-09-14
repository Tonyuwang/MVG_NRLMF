function [weight_v] = hsic_kernel_weights_p(Kernels_list,adjmat,dim,regcoef1,regcoef2)
%regcoef1=0.01
%regcoef2=0.001
%
%

num_kernels = size(Kernels_list,3);%����Kernels_list��ά�ȣ���ʵ������4ά

weight_v = zeros(num_kernels,1);%����num_kernels��һ�е�����󣬱�ʵ������4��һ��

y = adjmat;
    % Graph based kernel
if dim == 1
        ideal_kernel = y*y';%�γ�N*N�׵ľ���
else
        ideal_kernel = y'*y;
end
%ideal_kernel=Knormalized(ideal_kernel);

N_U = size(ideal_kernel,1);%����ideal_kernel���������
l=ones(N_U,1);
H = eye(N_U) - (l*l')/N_U;%eye(N_U)������ideal_kernel��������*�����ĵ�λ����
%eye(N_U)�����е�I��H�൱��������H��N_U�൱��N
M = zeros(num_kernels,num_kernels);%����4*4�������

for i=1:num_kernels %����˾���֮���������
	for j=1:num_kernels
		kk1 = H*Kernels_list(:,:,i)*H;
		kk2 = H*Kernels_list(:,:,j)*H;
		%kk1 = Kernels_list(:,:,i);
		%kk2 = Kernels_list(:,:,j);
		
		%mm = dot( a1,a2 )/( sqrt( sum( a1.*a1 ) ) * sqrt( sum( a2.*a2 ) ) );
		mm = trace(kk1'*kk2);
		m1 = trace(kk1*kk1');
		m2 = trace(kk2*kk2');
		M(i,j) = mm/(sqrt(m1)*sqrt(m2));%�����е�W
		%M(i,j) = (mm);
	end
end
%d_1 = sum(M);
%D_1 = diag(d_1);
%LapM = D_1 - M;
%d_tmep_1=eye(num_kernels)/(D_1^(1/2));
%LapM = d_tmep_1*L_D_1*d_tmep_1;


a = zeros(num_kernels,1);%����num_kernels��һ�е������

for i=1:num_kernels%����������tr��һ��

	kk = H*Kernels_list(:,:,i)*H;
	aa = trace(kk'*ideal_kernel);
	a(i) = aa*(N_U-1)^-2;
end

v = randn(num_kernels,1);%�������4��һ�еľ���
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