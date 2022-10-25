% Enroll new speakers that were not in the training data set.
% Create i-vectors for each file for each speaker in the enroll set using 
% the this sequence of steps:
%    1. Feature Extraction
%    2. Baum-Welch Statistics: Determine the zeroth and first order statistics
%    3. i-vector Extraction
%    4. Intrsession compensation
%    5. two-class and multi-class scoring
% Then average the i-vectors across files to create an i-vector model for 
% the speaker. Repeat the for each speaker.

clc;
clear all;

nworkers = 12;
nworkers = min(nworkers, feature('NumCores'));

% Load trained GMMs.
load('stats/svd_enroll_stats.mat');

% Set up SVD test data location
test_lists=dir(['lists/svd/test_data' '/*.lst']);

% load pathologies i-vectors
load('stats/pathologies_i_vectors.mat');

% Load the Total Variability matrix
load('stats/T_stats.mat');


n_pathologies = size(test_lists, 1);
numFeatures = 39;

% can be either "gplda" or "css"
scoring_method = "css";

if scoring_method == "gplda"
    % load gplda model
    load('stats/gpldaModel.mat');
end

spk_idx = 1;
counter = 1;

for ii=1:n_pathologies
    disp(['Testing i-vectors for speakers in ' test_lists(ii).name]);
    
    % set_list_file -> path of the list which holds the speakers ids used
    % for the speaker's GMMs.
    set_list_file = [test_lists(ii).folder '\' test_lists(ii).name];
    speakers_ids = strsplit(fileread(set_list_file))';
    n_speakers = size(speakers_ids, 1);

    Sigma = pathologies_gmm{ii}.sigma(:);
    TS = T_mats{ii}./Sigma;
    TSi = TS';
    
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
        f = features * gamma - n.*(pathologies_gmm{ii}.mu);
        
        %i-vector Extraction
        w = pinv(I + (TS.*repelem(n(:),numFeatures))' * T_mats{ii}) * TSi * f(:);

        % Intersession Compensation
        w = projectionMatrix*w;
        
        pathology_name{spk_idx} = session_name;
        ivectorPerSpeaker{spk_idx} = w;            
        
        % Score the speakers i-vector comparisson with every pathology's
        % i-vector using the gplda model or cosine similarity score
        if scoring_method == "gplda"
            for kk=1:n_pathologies
                scores(kk) = gpldaScore(gpldaModel,ivectorPerPathology{kk},ivectorPerSpeaker{spk_idx});         
            end    
        else 
            for kk=1:n_pathologies 
                scores(kk) = dot(ivectorPerSpeaker{spk_idx},...
                    ivectorPerPathology{kk})/(norm(ivectorPerPathology{kk})*norm(ivectorPerSpeaker{spk_idx}));
            end     
        end
        
        [max_num, max_id] = max(scores(:));
        pathology_found{spk_idx} = pathologies_gmm{max_id}.name;
        pathology_original{spk_idx} = pathologies_gmm{ii}.name;

        % Two-class score computation considering pathologies 4 and 5:
        if ii==4 || ii==5    
            [max_num, max_id] = max(scores(4:5));
            two_class_pathology{counter} = pathologies_gmm{3+max_id}.name;
            two_class_original{counter} = pathologies_gmm{ii}.name;
            counter = counter + 1;
        end
        
        spk_idx = spk_idx + 1;
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
labels_multi_class = {'Dysody', 'Dysphonia', 'Functional Dysphonia',...
    'Healthy', 'Hyper Functional Dysphonia', 'Hypo Functional Dysphonia',...
    'Spasmodic Dysphonia', 'Vocal Cyst Polyp'};
labels_two_class = {'Healthy', 'Hyper Functional Dysphonia'};

confusionchart(multi_class_CM,labels_multi_class);
%confusionchart(two_class_CM,labels_two_class);


