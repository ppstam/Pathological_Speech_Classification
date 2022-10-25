%% Calculation of total variability space for each pathology
clear;
close all;
nworkers = 12;
nworkers = min(nworkers, feature('NumCores'));

% Load the Baum-Welch statistics
load('stats/baum-welch_stats.mat');

% Set up SVD train data location
train_lists=dir(['lists/svd/train_data' '/*.lst']);

% Load trained GMMs.
load('stats/svd_enroll_stats.mat');

n_pathologies = size(train_lists, 1);
numFeatures = size(pathologies_gmm{1}.mu,1);
numComponents = size(pathologies_gmm{1}.mu,2);

% T matrix dimension % EM algorithm iterations
numTdim = 128;
numIterations = 20;

% Cell array to hold T matrix for each pathology GMM.
T_mats = cell(1,n_pathologies);

for ii=1:n_pathologies
    
    % set_list_file -> path of the list which holds the speakers ids used
    % for the speaker's GMMs.
    set_list_file = [train_lists(ii).folder '\' train_lists(ii).name];
    speakers_ids = strsplit(fileread(set_list_file))';
    n_speakers = size(speakers_ids, 1);
    
    Sigma = pathologies_gmm{ii}.sigma(:);
    
    % Initialize T and the identity matrix, and preallocate cell arrays.
    T = randn(numel(pathologies_gmm{ii}.sigma),numTdim);
    T = T/norm(T);
    
    I = eye(numTdim);
    
    Ey = cell(n_speakers,1);
    Eyy = cell(n_speakers,1);
    Linv = cell(n_speakers,1);
    
    N_stats = N{ii};
    F_stats = F{ii};
    Nc_i = Nc{ii};
    
    for iterIdx = 1:numIterations
        tic
    
        % 1. Calculate the posterior distribution of the hidden variable
        TtimesInverseSSdiag = (T./Sigma)';
        parfor s = 1:n_speakers
            L = (I + TtimesInverseSSdiag.*N_stats{s}*T);          
            Linv{s} = pinv(L);
            Ey{s} = Linv{s}*TtimesInverseSSdiag*F_stats{s};
            Eyy{s} = Linv{s} + Ey{s}*Ey{s}';
        end
    
        % 2. Accumlate statistics across the speakers
        Eymat = cat(2,Ey{:});
        FFmat = cat(2,F_stats{:});
        Kt = FFmat*Eymat';
        K = mat2cell(Kt',numTdim,repelem(numFeatures,numComponents));
    
        newT = cell(numComponents,1);
        for c = 1:numComponents
            AcLocal = zeros(numTdim);
            for s = 1:n_speakers
                AcLocal = AcLocal + Nc_i{s}(:,:,c)*Eyy{s};
            end
        
        % 3. Update the Total Variability Space
            newT{c} = (pinv(AcLocal)*K{c})';
        end
        T = cat(1,newT{:});

        fprintf('Training Total Variability Space for %s: %d/%d complete (%0.0f ms).\n',...
            pathologies_gmm{ii}.name,iterIdx,numIterations,toc * 1000)
    end
    T_mats{ii} =  T;
end

% We save the T_mats cell arrays for further use.
out_stats_file = 'stats/T_stats.mat';
disp(['Saving stats to ' out_stats_file]);
save(out_stats_file, 'I','T_mats');
