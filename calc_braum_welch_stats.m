%% Calculation of Baum-Welch statistics 
% zero (N) and first (F) order statistics used in the EM algorithm, 
%calculated using the final GMMs.

clear;
close all;
nworkers = 12;
nworkers = min(nworkers, feature('NumCores'));

% Set up SVD train data location
train_lists=dir(['lists/svd/train_data' '/*.lst']);

% Load trained GMMs.
load('stats/svd_enroll_stats.mat');

n_pathologies = size(train_lists, 1);
numFeatures = size(pathologies_gmm{1}.mu,1);
numComponents = size(pathologies_gmm{1}.mu,2);

% The Baum-Welch statistics are the N (zeroth order) and F (first order)
% statistics used in the EM algorithm, calculated using the final GMMs. 

N = cell(1, n_pathologies);
F = cell(1, n_pathologies);
Nc = cell(1, n_pathologies);
Fc = cell(1, n_pathologies);
for ii = 1:n_pathologies  
    
    disp(['Calculating Baum-Welch statistics for: ' pathologies_gmm{ii}.name]);
    % set_list_file -> path of the list which holds the speakers ids used
    % for the speaker's GMMs.
    set_list_file = [train_lists(ii).folder '\' train_lists(ii).name];
    speakers_ids = strsplit(fileread(set_list_file))';
    n_speakers = size(speakers_ids, 1);
    
    Nc_i = {};
    Fc_i = {};
    
    
    for jj = 1:n_speakers   
        session_name = [speakers_ids{jj} '.ascii'];
        features = load(session_name, '-ascii');
        features = features';
        
        % Compute a posteriori log-likelihood
        logLikelihood = helperGMMLogLikelihood(features,pathologies_gmm{ii});
        
        % Compute a posteriori normalized probability
        amax = max(logLikelihood,[],1);
        logLikelihoodSum = amax + log(sum(exp(logLikelihood-amax),1));
        gamma = exp(logLikelihood - logLikelihoodSum)';
        
        % Compute Baum-Welch statistics
        n = sum(gamma,1);
        f = features * gamma;
        
        Nc_i{jj} = reshape(n,1,1,numComponents);
        Fc_i{jj} = reshape(f,numFeatures,1,numComponents);
    end
    
    % Expand the statistics into matrices and center F(s), such that:
    % N(s) is a C F×C F diagonal matrix whose blocks are Nc(s)I (c=1,...C).
    % F(s) is a C F×1 supervector obtained by concatenating Fc(s)  (c=1,...C).
    % C is the number of components in the UBM.
    % F is the number of features in a feature vector.
    N_i = Nc_i;
    F_i = Fc_i;
    muc = reshape(pathologies_gmm{ii}.mu,numFeatures,1,[]);
    for s = 1:n_speakers
        N_i{s} = repelem(reshape(Nc_i{s},1,[]),numFeatures);
        F_i{s} = reshape(Fc_i{s} - Nc_i{s}.*muc,[],1);
    end
    
    Nc{ii} = Nc_i;
    Fc{ii} = Fc_i;
    N{ii} = N_i;
    F{ii} = F_i;
end

% We save the N,F cell arrays for further use.
out_stats_file = 'stats/baum-welch_stats.mat';
disp(['Saving stats to ' out_stats_file]);
save(out_stats_file, 'N','F','Nc','Fc');
