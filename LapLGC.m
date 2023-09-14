function [LapA] = LapLGC(W1,W2,y, lambda1,lambda2,p_nearest_neighbor)
%Graph-based Semi-supervised Learning with Local and Global Consistency基于图的局部和全局一致性半监督学习
%tju cs, bioinformatics. This program is recoded by reference follow:
%ref:
%[1]  Zha Z J, Mei T, Wang J, et al. 
%            Graph-based semi-supervised learning with multiple labels[J]. 
%                Journal of Visual Communication & Image Representation, 2009, 20(2):97-103.
%
%[2] Ghosh A, Sekhar C C. 
%             Label Correlation Propagation for Semi-supervised Multi-label Learning[J]. 2017. 
% 
%[3] Chen G, Song Y, Wang F, et al. 
%          Semi-supervised Multi-label Learning by Solving a Sylvester Equation[C]// 
%            Siam International Conference on Data Mining, SDM 2008, April 24-26, 2008, Atlanta, Georgia, Usa. DBLP, 2008:410-419.
%                
% W1 : the kernel of object 1, (m-by-m)
% W2 : the kernel of object 2, (n-by-n)
% y  : binary adjacency matrix, (m-by-n)
%lambda1: Regularized item for W1 (0.125)
%lambda2: Regularized item for W2 (0.125)
%p_nearest_neighbor: the p nearest neighbor samples (30)
%Network of Laplacian Regularized Least Square
[num_1,num_2] = size(y);
%1.Sparsification of the similarity matrices相似矩阵的稀疏化
%1.1
fprintf('Caculating nearest neighbor graph 1\n');
N_1 = neighborhood_Com(W1,p_nearest_neighbor);
fprintf('Sparsification of the similarity matrix 1\n');
S_1 = N_1.*W1;
d_1 = sum(S_1);
D_1 = diag(d_1);
L_D_1 = D_1 - S_1;
%Laplacian Regularized
d_tmep_1=eye(num_1)/(D_1^(1/2));
L_D_11 = d_tmep_1*L_D_1*d_tmep_1;

%1.2
fprintf('Caculating nearest neighbor graph 2\n');
N_2 = neighborhood_Com(W2,p_nearest_neighbor);
fprintf('Sparsification of the similarity matrix 2\n');
S_2 = N_2.*W2;
d_2 = sum(S_2);
D_2 = diag(d_2);
L_D_2 = D_2 - S_2;
%Laplacian Regularized
d_tmep_2=eye(num_2)/(D_2^(1/2));
L_D_22 = d_tmep_2*L_D_2*d_tmep_2;

%1.3 Solve Laplacian Regularized Label Propagation解决拉普拉斯正则化标签传播
fprintf('Laplacian Regularized Label Propagation\n');
LapA =[];

AA = (lambda1*L_D_11 + eye(num_1));
BB = lambda2*L_D_22;
CC = y;

%Solve Sylvester equation AX + XB = C for X
LapA = sylvester(AA,BB,CC);


end







function similarities_N = neighborhood_Com(similar_m,kk)

similarities_N=zeros(size(similar_m));

mm = size(similar_m,1);

for ii=1:mm
	
	for jj=ii:mm
		iu = similar_m(ii,:);
		iu_list = sort(iu,'descend');
		iu_nearest_list_end = iu_list(kk);
		
		ju = similar_m(:,jj);
		ju_list = sort(ju,'descend');
		ju_nearest_list_end = ju_list(kk);
		if similar_m(ii,jj)>=iu_nearest_list_end & similar_m(ii,jj)>=ju_nearest_list_end
			similarities_N(ii,jj) = 1;
			similarities_N(jj,ii) = 1;
		elseif similar_m(ii,jj)<iu_nearest_list_end & similar_m(ii,jj)<ju_nearest_list_end
			similarities_N(ii,jj) = 0;
			similarities_N(jj,ii) = 0;
		else
			similarities_N(ii,jj) = 0.5;
			similarities_N(jj,ii) = 0.5;
		end
	
	
	end


end

end
