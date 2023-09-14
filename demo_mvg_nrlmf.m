clear;
seed = 12345678;
rand('seed', seed);
nfolds = 10; nruns=1;

%dataname = 'nr';%alph = 1;p_nearest_neighbor=9; //MKL-2^-0,2^-1
%dataname = 'gpcr';%alph = 0.005;p_nearest_neighbor=10; //MKL-2^-1,2^-4
dataname = 'ic';%alph = 0.005;p_nearest_neighbor=10; //MKL-2^-0,2^-1
%dataname = 'e';%alph = 0.0001;p_nearest_neighbor=10; //MKL-2^-4,2^-0
load(['data/kernels/' dataname '_Drug_MACCS_fingerprint.mat']);
dataname
% load adjacency matrix
[Y,l1,l2] = loadtabfile(['data/interactions/' dataname '_admat_dgc.txt']);
%[data_r,data_c] = size(Y);

gamma=0.3;
gamma_fp = 1;

fold_aupr_nrlmf_ka=[];fold_auc_nrlmf_ka=[];
% alph = 1;                                                                
% k_nn = 6;
% kr1 = 50;kr2 = 110;k_p = 9;
lamda_1 = 0.00125;lamda_2 = 0.00125;Iteration_max = 300;k = 200;learn_rate = 0.02;
alpha_1 = 0.25;beta_2 = 0.15;p_nearest_neighbor = 5;c = 5;
globa_true_Y_lp=[];
globa_predict_Y_lp=[];
isHG=0;
% lambda1 = 1;lambda2=1;

for run=1:nruns
    % split folds
%     crossval_idx = crossvalind('Kfold', length(y(:)), nfolds);
    crossval_idx = crossvalind('Kfold',Y(:),nfolds);
      

    for fold=1:nfolds
        t1 = clock;
        train_idx = find(crossval_idx~=fold);
        test_idx  = find(crossval_idx==fold);
        
        Y_train = Y;
        Y_train(test_idx) = 0;

        %%  1.kernels
		%% load kernels
        k1_paths = {['data/kernels/' dataname '_simmat_proteins_sw-n.txt'],...
                    ['data/kernels/' dataname '_simmat_proteins_go.txt'],...
                    ['data/kernels/' dataname '_simmat_proteins_ppi.txt'],...
                    };
		K1 = [];
        for i=1:length(k1_paths)
            [mat, labels] = loadtabfile(k1_paths{i});
            mat = process_kernel(mat);
            K1(:,:,i) = Knormalized(mat);
        end
		
        %K1(:,:,i+1) = kernel_gip_0(y_train,1, gamma);
		K1(:,:,i+1) = getGipKernel(Y_train,gamma);
        k2_paths = {['data/kernels/' dataname '_simmat_drugs_simcomp.txt'],...
                   
                    ['data/kernels/' dataname '_simmat_drugs_sider.txt'],...
                    };
        K2 = [];
        for i=1:length(k2_paths)
            [mat, labels] = loadtabfile(k2_paths{i});
            mat = process_kernel(mat);
            K2(:,:,i) = Knormalized(mat);
        end
		K2(:,:,i+1) = kernel_gip_0(Drug_MACCS_fingerprint,1, gamma_fp);
        %K2(:,:,i+1+1) = kernel_gip_0(y_train,2, gamma);
		K2(:,:,i+1+1) = getGipKernel(Y_train',gamma);
		
        %% perform predictions
        %lambda = 1;
     	% 2. multiple kernel 
%  		weights_1;
% %     weight_v1 = 1/size(K1,3);
%  		weight_2;


		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		
		% 2. multiple kernel 
		
		
		
		%[weight_v1] = hsic_kernel_weights_p(K1,Y_train,1,-2^-1,2^-4);
		%K_COM1 = combine_kernels(weight_v1, K1);
		
		
		
		%[weight_v2] = hsic_kernel_weights_p(K2,Y_train,2,-2^-1,2^-4);
		%K_COM2 = combine_kernels(weight_v2, K2);
        
		
		
		%3.DLapRLS
		[A_cos_com] = mvg_nrlmf(K1,K2,Y_train,k,Iteration_max,gamma,lamda_1,lamda_2,c,alpha_1,beta_2,learn_rate,p_nearest_neighbor,1,isHG);

		
		t2=clock;
		etime(t2,t1)
		
		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		
        %% 4. evaluate predictions
        YY=Y;
        %yy(yy==0)=-1;
        %stats = evaluate_performance(y2(test_idx),yy(test_idx),'classification');
		test_labels = YY(test_idx);
		predict_scores = A_cos_com(test_idx);
		[X,Z,tpr,aupr_nrlmf_A_KA] = perfcurve(test_labels,predict_scores,1, 'xCrit', 'reca', 'yCrit', 'prec');
		
		[X,Z,THRE,AUC_nrlmf_KA,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(test_labels,predict_scores,1);

		
		fprintf('---------------\nRUN %d - FOLD %d  \n', run, fold)

		fprintf('%d - FOLD %d - weighted_kernels_nrlmf_AUPR: %f \n', run, fold, aupr_nrlmf_A_KA)
		

		fold_aupr_nrlmf_ka=[fold_aupr_nrlmf_ka;aupr_nrlmf_A_KA];
		fold_auc_nrlmf_ka=[fold_auc_nrlmf_ka;AUC_nrlmf_KA];

		
		globa_true_Y_lp=[globa_true_Y_lp;test_labels];
		globa_predict_Y_lp=[globa_predict_Y_lp;predict_scores];

		
    end
    
    
end
RMSE = sqrt(sum((globa_predict_Y_lp-globa_true_Y_lp).^2)/length(globa_predict_Y_lp))



mean_aupr_kronls_ka = mean(fold_aupr_nrlmf_ka)
mean_auc_kronls_ka = mean(fold_auc_nrlmf_ka)
