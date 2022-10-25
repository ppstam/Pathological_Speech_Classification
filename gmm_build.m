% Building Gmms for SVD Pathologies.
% We use the train data lists created in prepareSubsets.m to build a GMM
% for every pathology and Healthy.

clear;
close all;
addpath ./functions
nworkers = 12;
nworkers = min(nworkers, feature('NumCores'));
%% Training the UBM from the training data of SVD
% Set up SVD train data location
train_lists=dir(['lists/svd/train_data' '/*.lst']);

n_pathologies = size(train_lists, 1);
pathologies_gmm = cell(n_pathologies, 1);
nmix = 2; %number of mixtures 
final_niter = 100;
ds_factor = 1;

% We create every pathology's GMM in a for loop.
tic
for i = 1:n_pathologies
    disp(['Creating GMM for ' train_lists(i).name]);
    
    % set_list_file -> path of the list which holds the speakers ids used
    % for the pathology's GMM.
    %speaker ids = wav_ids
    set_list_file = [train_lists(i).folder '\' train_lists(i).name];
    wav_ids = strsplit(fileread(set_list_file))';
    n_wavs = size(wav_ids, 1);

    % We collect all the mfccs from all the wavs used for the
    % pathology's GMM creation in a single array. Then we use this array
    % to extract zero and first order statistics.
    features = [];
    for ii = 1:n_wavs
        session_name = [wav_ids{ii} '.ascii'];
        f = load(session_name, '-ascii');
        features = [features f'];      
        mfcc_mat{ii,1}=f';
    end
    mfccs{i,1}=mfcc_mat;
    pathology_gmm = gmm_em(mfccs{i,1}, nmix, final_niter, ds_factor, nworkers);
        
    pathology_gmm.name = extractBefore(train_lists(i).name, '.'); 

    pathologies_gmm{i,1} = pathology_gmm;
end
% We save the pathologies_gmm cell array for further use in the test phase.
out_stats_file = 'stats/svd_enroll_stats.mat';
disp(['Saving enrollment stats to ' out_stats_file]);
save(out_stats_file, 'pathologies_gmm');