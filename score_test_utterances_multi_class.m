% We use the test data lists created in prepareSubsets.m to score the
% feature vectors of each test utterance
clc;
clear all;
addpath ./functions
%%
% Load trained GMMs.
load('stats/svd_enroll_stats.mat');

% Set up SVD test data location
test_lists=dir(['lists/svd/test_data' '/*.lst']);

n_pathologies = size(test_lists, 1);
% Initialize the Confusion Matrix
CM=zeros(n_pathologies,n_pathologies);

%map adaptation
ADAPT=1; % if 1 does map adaptation; if 0 it does not map adaptation;
map_tau = 16.0; %relevance factor between 8 and 20; default 16
config = ['m','v','w'];

tic
if (ADAPT)
    for i = 1:n_pathologies 
       disp(['adapting trained GMM for ' test_lists(i).name]);
        
       % set_list_file -> path of the list which holds the wav ids used
       % for each pathology.
        set_list_file = [test_lists(i).folder '\' test_lists(i).name];
        utterances_ids = strsplit(fileread(set_list_file))';
        n_utterances = size(utterances_ids, 1);
    
        for ii = 1:n_utterances
            session_name = [utterances_ids{ii} '.ascii'];
            f = load(session_name, '-ascii');
            mfcc_mat{ii,1}=f';
        end
        mfccs{i,1}=mfcc_mat;
    
        %trained GMM
        tgmm.w=pathologies_gmm{i,1}.w;
        tgmm.mu=pathologies_gmm{i,1}.mu;
        tgmm.sigma=pathologies_gmm{i,1}.sigma;
        map_gmm{i,1} = mapAdapt(mfccs{i,1}, tgmm, map_tau, config);
    end
end
%    For each pathology
idx = 1;
for i = 1:n_pathologies
    disp(['Scoring utterances from ' test_lists(i).name]);
        
    % set_list_file -> path of the list which holds the speakers ids used
    % for the speaker's GMMs.
    set_list_file = [test_lists(i).folder '\' test_lists(i).name];
    utterances_ids = strsplit(fileread(set_list_file))';
    n_utterances = size(utterances_ids, 1);
    
    for ii = 1:n_utterances
        session_name = [utterances_ids{ii} '.ascii'];
        f = load(session_name, '-ascii');
                         
        %Score each test utterance against all pathologies map-adapted gmms
        emax=-1e06; % define a negative number 
        for jj=1:n_pathologies % predicted pathology
            %fetch gmm to compute likelihood
            if (ADAPT)
                tgmm.ComponentProportion=map_gmm{jj,1}.w;
                tgmm.mu=map_gmm{jj,1}.mu;
                tgmm.sigma=map_gmm{jj,1}.sigma;
            else
                tgmm.ComponentProportion=pathologies_gmm{jj,1}.w;
                tgmm.mu=pathologies_gmm{jj,1}.mu;
                tgmm.sigma=pathologies_gmm{jj,1}.sigma;
            end   
            pathology_logLikelihood = helperGMMLogLikelihood(f',tgmm);
            amax=max(pathology_logLikelihood,[],1);
            logLikelihoodSum=amax+log(sum(exp(pathology_logLikelihood-amax),1));
            L_sum=sum(logLikelihoodSum);
            %L_pathology = helperLogSumExp(pathology_logLikelihood);
            %L_sum=sum(L_pathology);
            if (L_sum>emax)
                emax=L_sum;
                ind_max=jj;
            end
            pathology_found{idx} = pathologies_gmm{ind_max}.name;
            pathology_original{idx} = pathologies_gmm{i}.name;
        end
        idx = idx+1;
    end
end
disp('Multi-class confusion matrix:');
multi_class_CM = confusionmat(pathology_original,pathology_found);
disp(multi_class_CM);
test_utterances_sum = sum(sum(multi_class_CM));
acc = trace(multi_class_CM) / test_utterances_sum;
disp(['Multi-class accuracy= ' num2str(acc)]);

%% Plot confusion matrices
% SVD classes labels
labels_multi_class = {'Dysody', 'Dysphonia', 'Functional Dysphonia',...
    'Healthy', 'Hyper Functional Dysphonia', 'Hypo Functional Dysphonia',...
    'Spasmodic Dysphonia', 'Vocal Cyst Polyp'};
labels_two_class = {'Healthy', 'Hyper Functional Dysphonia'};

confusionchart(multi_class_CM,labels_multi_class);
