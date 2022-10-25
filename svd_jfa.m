%% Setup files
clc;
clear all;
addpath ./functions

% Set up SVD train data location
train_lists=dir(['lists/svd/train_data' '/*.lst']);

% Set up SVD test data location
test_lists=dir(['lists/svd/test_data' '/*.lst']);

n_pathologies = size(train_lists, 1);
pathologies_gmm = cell(n_pathologies, 1);

% We load the jfa's ubm parameters as supervectors
m=load('models/ubm_means')';
v=load('models/ubm_variances')';
w=load('models/ubm_weights')';

n_mixtures  = size(w, 1);
dim         = size(m, 1) / n_mixtures;

% We load the model as superverctors, so we reshape 
% it to have each gaussian in one column
m = reshape(m, dim, n_mixtures);
v = reshape(v, dim, n_mixtures);

% We create a cell array representing jfa's ubm
jfa.mu = m;
jfa.sigma = v;
jfa.w = w';

%% MAP adaptation
% MAP adaptation in SVD Pathologies' GMMs using data from jfa cookbook.
% We use the train data lists created in prepareSubsets.m to create a GMM
% for every pathology (considering the 'Healthy' folder of SVD as another 
% "pathology").

% Set the relevance factor for map adaptation
relevanceFactor = 16;

% We create every pathology's GMM in a for loop.
tic
for i = 1:n_pathologies
    disp(['Creating GMM for ' train_lists(i).name]);
    
    % set_list_file -> path of the list which holds the speakers ids used
    % for the pathology's GMM.
    set_list_file = [train_lists(i).folder '\' train_lists(i).name];
    speaker_ids = strsplit(fileread(set_list_file))';
    n_speakers = size(speaker_ids, 1);

    % We collect all the mfccs from all the speakers used for the
    % pathology's GMM creation in a single array. Then we use this array
    % to extract zero and first order statistics.
    features = [];
    for ii = 1:n_speakers
        session_name = [speaker_ids{ii} '.ascii'];
        f = load(session_name, '-ascii');
        features = [features f'];         
    end
    
    % Extract zero (N) and first (F) order statistics using the features
    % array and the jfa ubm.
    [N, F, S] = helperExpectation(features, jfa);
        
    % Determine the maximum likelihood
    pathology_gmm = helperMaximization(N,F);

    % Determine adaption coefficient
    alpha = N ./ (N + relevanceFactor);

    % Adapt GMM mean. The covariances and weights of the speaker's GMM
    % remain the same as the jfa's ubm used for the adaptation.
    pathology_gmm.mu =  alpha.*pathology_gmm.mu + (1-alpha).*m;
    pathology_gmm.sigma = jfa.sigma;
    pathology_gmm.w = jfa.w;
    
    pathology_gmm.name = extractBefore(train_lists(i).name, '.'); 

    pathologies_gmm{i} = pathology_gmm;
end
fprintf('Enrollment completed in %0.2f seconds.\n',toc);
%% Evaluation
% n_speakers -> the number of speakers to test.
% pathology_found -> cell array holding the pathology found after the
% loglikelihoods comparison.
% pathology_original -> cell array holding the original pathology of the
% speaker tested.

% Counter used for the two-class and multi-class scoring computations below
idx = 1;
idx2 = 1;


tic
for i = 1:n_pathologies
    % set_list_file -> path of the list which holds the speakers ids used
    % for the speaker's mfccs.
    set_list_file = [test_lists(i).folder '\' test_lists(i).name];
    speakers_ids = strsplit(fileread(set_list_file))';
    n_speakers = size(speakers_ids, 1);
    
    pathology = extractBefore(test_lists(i).name, '.');
    disp(['Testing speakers for ' pathology]);
    
    for ii = 1:n_speakers
        session_name = [speakers_ids{ii} '.ascii'];
        features = load(session_name, '-ascii');
        features = features';
        
        % We compute each pathology's gmm log likelihood given the n-th's speaker's features.
        for j = 1:n_pathologies
            pathology_logLikelihood = helperGMMLogLikelihood(features,pathologies_gmm{j});
            L_pathology = helperLogSumExp(pathology_logLikelihood);
            L_pathologies(j) = sum(L_pathology);
        end

        % We assign to the speaker the pathology with the maximum probability
        % (pathology_found). We also keep speaker's original pathology for
        % score computation.
        [max_num, max_id] = max(L_pathologies(:));
        pathology_found{idx2} = pathologies_gmm{max_id}.name;
        pathology_original{idx2} = pathology;
    
        % Two-class score computation considering pathologies 4 and 5:
        if pathology == "Healthy" || pathology == "Hyper Functional Dysphonia"    
            [max_num, max_id] = max(L_pathologies(4:5));
            two_class_pathology{idx} = pathologies_gmm{3+max_id}.name;
            two_class_original{idx} = pathology;
            idx = idx + 1;
        end   
        idx2 = idx2 + 1;
    end
end

% Compute confusion matrix and accurasy for the multi-class problem
disp('Multi-class confusion matrix:');
multi_class_CM = confusionmat(pathology_original,pathology_found);
disp(multi_class_CM);
test_utterances_sum = sum(sum(multi_class_CM));
acc = trace(multi_class_CM) / test_utterances_sum;
disp(['Multi-class accuracy= ' num2str(acc)]);


% Compute confusion matrix and accurasy for the two-class problem
disp('Two-class confusion matrix (pathologies 4 and 5):');
two_class_CM = confusionmat(two_class_original,two_class_pathology);
disp(two_class_CM);
test_utterances_sum = sum(sum(two_class_CM));
acc = trace(two_class_CM) / test_utterances_sum;
disp(['Two-class accuracy= ' num2str(acc)]);

%% Plot confusion matrices
%Un-comment to use

labels_multi_class = {'Dysody', 'Dysphonia', 'Functional Dysphonia',...
    'Healthy', 'Hyper Functional Dysphonia', 'Hypo Functional Dysphonia',...
    'Spasmodic Dysphonia', 'Vocal Cyst Polyp'};
labels_two_class = {'Healthy', 'Hyper Functional Dysphonia'};


confusionchart(multi_class_CM,labels_multi_class);

%un-comment to use
%confusionchart(two_class_CM,labels_two_class);
