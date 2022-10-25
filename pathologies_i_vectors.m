%% i-vectors extraction for speakers in the test sets.

clc;
clear all;
nworkers = 12;
addpath ./functions
nworkers = min(nworkers, feature('NumCores'));

% Set up SVD train data location
train_lists=dir(['lists/svd/train_data' '/*.lst']);

% Load trained GMMs.
load('stats/svd_enroll_stats.mat');

% Load the Total Variability matrix
load('stats/T_stats.mat');

n_pathologies = size(train_lists, 1);
numFeatures = 39;

performLDA = 1;
performWCCN = 1;

for ii=1:n_pathologies
    disp(['Extracting i-vectors for ' train_lists(ii).name]);
    
    % set_list_file -> path of the list which holds the speakers ids used
    % for the speaker's GMMs.
    set_list_file = [train_lists(ii).folder '\' train_lists(ii).name];
    speakers_ids = strsplit(fileread(set_list_file))';
    n_speakers = size(speakers_ids, 1);

    Sigma = pathologies_gmm{ii}.sigma(:);
    TS = T_mats{ii}./Sigma;
    TSi = TS';
    
    for jj = 1:n_speakers
        session_name = [speakers_ids{jj} '.ascii'];
        features = load(session_name, '-ascii');
        features = features';
    end
        
        % Compute a posteriori log-likelihood
        logLikelihood = helperGMMLogLikelihood(features,pathologies_gmm{ii});
        
        % Compute a posteriori normalized probability
        amax = max(logLikelihood,[],1);
        logLikelihoodSum = amax + log(sum(exp(logLikelihood-amax),1));
        gamma = exp(logLikelihood - logLikelihoodSum)';
        
        % Compute Baum-Welch statistics
        n = sum(gamma,1);
        f = features * gamma - n.*(pathologies_gmm{ii}.mu);
        
        ivectorPerPathology{ii} = pinv(I + (TS.*repelem(n(:),numFeatures))' * T_mats{ii}) * TSi * f(:);
end


%% LDA
% LDA attempts to minimize the intra-class variance and 
% maximize the variance between speakers

% Create a matrix of the training vectors and a map indicating which 
% i-vector corresponds to which speaker. Initialize the projection matrix 
% as an identity matrix.

w = ivectorPerPathology;
utterancePerSpeaker = cellfun(@(x)size(x,2),w);

ivectorsTrain = cat(2,w{:});
projectionMatrix = eye(size(w{1},1));

if performLDA == 1
   tic
    numEigenvectors = 128;

    Sw = zeros(size(projectionMatrix,1));
    Sb = zeros(size(projectionMatrix,1));
    wbar = mean(cat(2,w{:}),2);
    for ii = 1:numel(w)
        ws = w{ii};
        wsbar = mean(ws,2);
        Sb = Sb + (wsbar - wbar)*(wsbar - wbar)';
        Sw = Sw + cov(ws',1);
    end
    
    [A,~] = eigs(Sb,Sw,numEigenvectors);
    A = (A./vecnorm(A))';

    ivectorsTrain = A * ivectorsTrain;
    
    w = mat2cell(ivectorsTrain,size(ivectorsTrain,1),utterancePerSpeaker);

    projectionMatrix = A * projectionMatrix;

    fprintf('LDA projection matrix calculated (%0.2f seconds).\n',toc) 
end

%% WCCN

if performWCCN == 1
   tic
    alpha = 0.9;
    
    W = zeros(size(projectionMatrix,1));
    for ii = 1:numel(w)
        W = W + cov(w{ii}',1);
    end
    W = W/numel(w);
    
    W = (1 - alpha)*W + alpha*eye(size(W,1));

    B = chol(pinv(W),'lower');
    
    projectionMatrix = B * projectionMatrix;
    
    fprintf('WCCN projection matrix calculated (%0.4f seconds).\n',toc) 
end


% We save the N,F cell arrays for further use.
out_stats_file = 'stats/pathologies_i_vectors.mat';
disp(['Saving stats to ' out_stats_file]);
save(out_stats_file, 'ivectorPerPathology','projectionMatrix');
