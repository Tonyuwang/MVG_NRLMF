function [F] = mvg_nrlmf(W1,W2,Y,k,Iteration_max,gamma,lamda_1,lamda_2,c,alpha_1,beta_2,learn_rate,p_nearest_neighbor,pre,isHG)
%tju cs, bioinformatics. This program is coded by reference follow:
%ref:
%[1] Liu Y, Wu M, Miao C, et al. 
%    Neighborhood Regularized Logistic Matrix Factorization for Drug-Target Interaction Prediction.[J]. 
%    Plos Computational Biology, 2016, 12(2):e1004760.
%
%
%


%Neighborhood Regularized Logistic Matrix Factorization (LMF)
%This program is used to Collaborative filtering. 
% W1 : the kernel of object 1, (m-by-m)
% W2 : the kernel of object 2, (n-by-n)
% Y  : binary adjacency matrix, (m-by-n)
% k  : the k is the dimension of the feature spaces
% Iteration_max  : the Iteration_maxis the max numbers of Iteration
%lamda_1 (0.125), lamda_2 (0.125), alpha_1 (0.0001) beta_2(0.0001) : the are regularization coefficients of kernel W1, kernel W2, U, V.
%c (3) 
%learn_rate: the learn rate of Gradient decline (0.001)

F=[];
fprintf('Neighborhood Regularized Logistic Matrix Factorization\n'); 
[num_1,num_2] = size(Y);%y的几行几列
L1 = zeros(size(W1));%生成四行四
L2 = zeros(size(W2));
%0 preprocesses the interaction matrix Y 
    if pre==1
	fprintf('Y preprocesses\n');
    Y = preprocess(W1,W2,Y);   
else
   fprintf('No preprocesses\n');
    end
%1.Sparsification of the similarity matrices相似矩阵的稀疏化
%1.1
fprintf('Caculating nearest neighbor graph 1\n');

for i=1:size(L1,3)
	if isHG==1
		[L_D_11] = construct_Hypergraphs_knn(W1(:,:,i),p_nearest_neighbor);
	
	else
	
		N_1 = neighborhood_Com(W1(:,:,i),p_nearest_neighbor);

		S_1 = N_1.*W1(:,:,i);
		%S_1 = W1(:,:,i);
		d_1 = sum(S_1);
		D_1 = diag(d_1);
		L_D_1 = D_1 - S_1;
		%Laplacian Regularized
		d_tmep_1=eye(num_1)/(D_1^(1/2));
		L_D_11 = d_tmep_1*L_D_1*d_tmep_1;
	
	end

L1(:,:,i) = L_D_11;
end
%1.2
fprintf('Caculating nearest neighbor graph 2\n');
for i=1:size(L2,3)
	if isHG==1
		[L_D_22] = construct_Hypergraphs_knn(W2(:,:,i),p_nearest_neighbor);
	else
		N_2 = neighborhood_Com(W2(:,:,i),p_nearest_neighbor);
		S_2 = N_2.*W2(:,:,i);
		%S_2 =W2(:,:,i);
		d_2 = sum(S_2);
		D_2 = diag(d_2);
		L_D_2 = D_2 - S_2;
		%Laplacian Regularized
		d_tmep_2=eye(num_2)/(D_2^(1/2));
		L_D_22 = d_tmep_2*L_D_2*d_tmep_2;
	end

L2(:,:,i) = L_D_22;
end

%1.3 Solve Laplacian Regularized Label Propagation解决拉普拉斯正则化标签传播
fprintf('Multi-view Laplacian Regularized Label Propagation\n');
weights_1 = 1/size(L1,3);%Av,t
weights_2 = 1/size(L2,3);%Av,t
%for i=1:Iteration
%    L_D_11_Com = combine_kernels(weights_1.^gamma, L1);%Ld*
%    L_D_22_Com = combine_kernels(weights_2.^gamma, L2);%Lt*

  % AA = (lambda1*L_D_11_Com + eye(num_1));
   %BB = lambda2*L_D_22_Com;
   %CC = Y;

   %Solve Sylvester equation AX + XB = C for X
   %LapA = sylvester(AA,BB,CC);

%   weights_1=computing_weights(LapA,L1,gamma,1);
%   weights_2=computing_weights(LapA,L2,gamma,2);
%end

%%2 Regularized Logistic Matrix Factorization正则Logistic矩阵分解
[m,n]=size(Y);
%initial value of U and V 
%U = normrnd(0,1/sqrt(k),m,k);  %(m-by-k)
%V = normrnd(0,1/sqrt(k),k,n);  %(k-by-n)
%V = V';

[U1,S_k,V1] = svds(Y,k);
U = U1*(S_k^0.5);  %(m-by-k)
V = V1*(S_k^0.5);  %(n-by-k)

[P] = sigm_v(U*V');


%objective function:

%sloving the problem by Gradient decline
for i=1:Iteration_max 
	 L_D_11_Com = combine_kernels(weights_1.^gamma, L1);%Lt*
    L_D_22_Com = combine_kernels(weights_2.^gamma, L2);%Ld*
	
	delta_U = P*V + (c - 1)*(Y.*P)*V - c*Y*V + (lamda_1*eye(m) + alpha_1*L_D_11_Com)*U;
	delta_V = P'*U + (c - 1)*(Y'.*P')*U - c*Y'*U + (lamda_2*eye(n) + beta_2*L_D_22_Com)*V;
	U = U - learn_rate*delta_U;
	V = V - learn_rate*delta_V;
	

	[P] = sigm_v(U*V'); 
	weights_1=computing_weights(U,L1,gamma,1);
    weights_2=computing_weights(V,L2,gamma,1);
end


%reconstruct Y*
F = U*V';

[F] = sigm_v(F);


end



function norm_a = sigm_v(a)

	norm_a = (exp(a))./(1 + exp(a));

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


function newY=preprocess_Y(Y,Sd,St,K,eta)
%preprocesses the interaction matrix Y by replacing each
%of the 0's (i.e. presumed non-interactions) with a continuous value
%between 0 and 1. For each 0, the K nearest known drugs are used to infer
%a value, the K nearest known targets are used to infer another value, and
%then the average of the two values is used to replace that 0.
 % decay values to be used in weighting similarities later
    eta = eta .^ (0:K-1);
	newY =[];
    y2_new1 = zeros(size(Y));
    y2_new2 = zeros(size(Y));

    empty_rows = find(any(Y,2) == 0);   % get indices of empty rows
    empty_cols = find(any(Y)   == 0);   % get indices of empty columns

    % for each drug i...
    for i=1:length(Sd)
        drug_sim = Sd(i,:); % get similarities of drug i to other drugs
        drug_sim(i) = 0;    % set self-similiraty to ZERO

        indices  = 1:length(Sd);    % ignore similarities 
        drug_sim(empty_rows) = [];  % to drugs of 
        indices(empty_rows) = [];   % empty rows

        [~,indx] = sort(drug_sim,'descend');    % sort descendingly
        indx = indx(1:K);       % keep only similarities of K nearest neighbors
        indx = indices(indx);   % and their indices

        % computed profile of drug i by using its similarities to its K
        % nearest neighbors weighted by the decay values from eta
        drug_sim = Sd(i,:);
        y2_new1(i,:) = (eta .* drug_sim(indx)) * Y(indx,:) ./ sum(drug_sim(indx));
    end

    % for each target j...
    for j=1:length(St)
        target_sim = St(j,:); % get similarities of target j to other targets
        target_sim(j) = 0;    % set self-similiraty to ZERO

        indices  = 1:length(St);        % ignore similarities 
        target_sim(empty_cols) = [];    % to targets of
        indices(empty_cols) = [];       % empty columns

        [~,indx] = sort(target_sim,'descend');  % sort descendingly
        indx = indx(1:K);       % keep only similarities of K nearest neighbors
        indx = indices(indx);   % and their indices

        % computed profile of target j by using its similarities to its K
        % nearest neighbors weighted by the decay values from eta
        target_sim = St(j,:);
        y2_new2(:,j) = Y(:,indx) * (eta .* target_sim(indx))' ./ sum(target_sim(indx));
    end

    % average computed values of the modified 0's from the drug and target
    % sides while preserving the 1's that were already in Y 
    newY = max(Y,(y2_new1 + y2_new2)/2);
	

end

function weights=computing_weights(F,L_l,gamma,dim)

w = zeros(size(L_l,3),1);
weights = w;
e = 1/(gamma - 1);
	for i=1:length(w)
		if dim ==1
			d = F'*L_l(:,:,i)*F;
		else
			d = F*L_l(:,:,i)*F';
		end
		s = (1/trace(d))^e;
		w(i) = s;
	end
	for i=1:length(w)
		weights(i) = w(i)/(sum(w));
	end

end


function y_P = preprocess(W1,W2,y)
	w_v1 = randn(size(W1,3),1);
	w_v1(1:size(W1,3)) = 1/size(W1,3);
	w_v2 = w_v1;
	K_COM1 = combine_kernels(w_v1, W1);
	K_COM2 = combine_kernels(w_v2, W2);
	[y_P] = wknkn(y,K_COM1,K_COM2,16,0.9);

end

function [F_new] = wknkn(Y,W1,W2,k_nn,eta_v)
%tju cs, bioinformatics. This program is recoded by reference follow:
%Weighted K nearest known neighbors (WKNKN)
%ref:
%      Ezzat A, Zhao P, Wu M, et al. 
%      Drug-Target Interaction Prediction with Graph Regularized Matrix Factorization[J]. 
%           IEEE/ACM Transactions on Computational Biology & Bioinformatics, 2016, PP(99):1-1.
% W1 : the kernel of object 1, (m-by-m)
% W2 : the kernel of object 2, (n-by-n)
% Y  : binary adjacency matrix, (m-by-n)
%eta_v: decay term (<1)
%k_nn: the k nearest neighbor samples (30)

%initialize two matrices
[row_s col_s] = size(Y);
Y_d = zeros(row_s, col_s);
Y_t=Y_d;

for d=1:row_s
	dnn = KNearestKnownNeighbors(d,W1,k_nn);
	w_i = zeros(1,k_nn);
	for ii=1:k_nn
		w_i(ii) = (eta_v^(ii-1))*W1(d,dnn(ii));
	end
	%normalization term
	Z_d = [];
	Z_d = W1(d,dnn);
	Z_d = 1/(sum(Z_d));
	
	Y_d(d,:) = Z_d*(w_i*Y(dnn,:));

end


for t = 1:col_s
	tnn =  KNearestKnownNeighbors(t,W2,k_nn);
	w_j = zeros(1,k_nn);
	
	for jj=1:k_nn
		w_j(jj) = (eta_v^(jj-1))*W2(t,tnn(jj));
	end
	%normalization term
	Z_t = [];
	Z_t = W2(t,tnn);
	Z_t = 1/(sum(Z_t));
	
	Y_t(:,t) = Z_t*(Y(:,tnn)*w_j');
	
end

Y_dt = (Y_d+Y_t)/2;

F_new = [];
F_new = max(Y,Y_dt);

end


function similarities_N = KNearestKnownNeighbors(index_i,similar_m,kk)


		iu = similar_m(index_i,:);
		[B, iu_list] = sort(iu,'descend');
		similarities_N = iu_list(1:kk);

end

