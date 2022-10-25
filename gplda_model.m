%% GPLDA

clc;
clear all;

load('stats/pathologies_i_vectors.mat');

ivectors = cellfun(@(x)projectionMatrix*x,ivectorPerPathology,'UniformOutput',false);

numEigenVoices = 128;

%Determine the number of disjoint persons, the number of dimensions in the feature vectors,
%and the number of utterances per speaker.
K = numel(ivectors);
D = size(ivectors{1},1);
utterancePerSpeaker = cellfun(@(x)size(x,2),ivectors);

ivectorsMatrix = cat(2,ivectors{:});
N = size(ivectorsMatrix,2);
mu = mean(ivectorsMatrix,2);

ivectorsMatrix = ivectorsMatrix - mu;

% Determine a whitening matrix from the training i-vectors and then whiten 
% the i-vectors. Specify either ZCA whitening, PCA whitening, or no whitening.
whiteningType = ' ';

if strcmpi(whiteningType,'ZCA')
    S = cov(ivectorsMatrix');
    [~,sD,sV] = svd(S);
    W = diag(1./(sqrt(diag(sD)) + eps))*sV';
    ivectorsMatrix = W * ivectorsMatrix;
elseif strcmpi(whiteningType,'PCA')
    S = cov(ivectorsMatrix');
    [sV,sD] = eig(S);
    W = diag(1./(sqrt(diag(sD)) + eps))*sV';
    ivectorsMatrix = W * ivectorsMatrix;
else
    W = eye(size(ivectorsMatrix,1));
end


% Apply length normalization and then convert the training i-vector matrix back to a cell array.
ivectorsMatrix = ivectorsMatrix./vecnorm(ivectorsMatrix);

% Compute the global second-order moment:
S = ivectorsMatrix*ivectorsMatrix';

% Convert the training i-vector matrix back to a cell array.
ivectors = mat2cell(ivectorsMatrix,D,utterancePerSpeaker);

%Sort persons according to the number of samples and then group the i-vectors by 
%number of utterances per speaker. Precalculate the first-order moment of the i-th person
uniqueLengths = unique(utterancePerSpeaker);
numUniqueLengths = numel(uniqueLengths);

speakerIdx = 1;
f = zeros(D,K);
for uniqueLengthIdx = 1:numUniqueLengths
    idx = find(utterancePerSpeaker==uniqueLengths(uniqueLengthIdx));
    temp = {};
    for speakerIdxWithinUniqueLength = 1:numel(idx)
        rho = ivectors(idx(speakerIdxWithinUniqueLength));
        temp = [temp;rho]; %#ok<AGROW>

        f(:,speakerIdx) = sum(rho{:},2);
        speakerIdx = speakerIdx+1;
    end
    ivectorsSorted{uniqueLengthIdx} = temp; %#ok<SAGROW> 
end


%Initialize the eigenvoices matrix, V, and the inverse noise variance term, Λ.
V = randn(D,numEigenVoices);
Lambda = pinv(S/N);

%Specify the number of iterations for the EM algorithm and whether or not to apply the minimum divergence.
numIter = 200;
minimumDivergence = true;

%Train the G-PLDA model using the EM algorithm.
for iter = 1:numIter
    fprintf('\tIteration %d\n',iter)
    % EXPECTATION
    gamma = zeros(numEigenVoices,numEigenVoices);
    EyTotal = zeros(numEigenVoices,K);
    R = zeros(numEigenVoices,numEigenVoices);
    
    idx = 1;
    for lengthIndex = 1:numUniqueLengths
        ivectorLength = uniqueLengths(lengthIndex);
        
        % Isolate i-vectors of the same given length
        iv = ivectorsSorted{lengthIndex};
        
        % Calculate M
        M = pinv(ivectorLength*(V'*(Lambda*V)) + eye(numEigenVoices)); % Equation (A.7) in [13]
        
        % Loop over each speaker for the current i-vector length
        for speakerIndex = 1:numel(iv)
            
            % First moment of latent variable for V
            Ey = M*V'*Lambda*f(:,idx); % Equation (A.8) in [13]
            
            % Calculate second moment.
            Eyy = Ey * Ey';
            
            % Update Ryy 
            R = R + ivectorLength*(M + Eyy); % Equation (A.13) in [13]
            
            % Append EyTotal
            EyTotal(:,idx) = Ey;
            idx = idx + 1;
            
            % If using minimum divergence, update gamma.
            if minimumDivergence
                gamma = gamma + (M + Eyy); % Equation (A.18) in [13]
            end
        end
    end
    
    % Calculate T
    TT = EyTotal*f'; % Equation (A.12) in [13]
    
    % MAXIMIZATION
    V = TT'*pinv(R); % Equation (A.16) in [13]
    Lambda = pinv((S - V*TT)/N); % Equation (A.17) in [13]

    % MINIMUM DIVERGENCE
    if minimumDivergence
        gamma = gamma/K; % Equation (A.18) in [13]
        V = V*chol(gamma,'lower');% Equation (A.22) in [13]
    end
end

gpldaModel = struct('mu',mu, ...
                    'WhiteningMatrix',W, ...
                    'EigenVoices',V, ...
                    'Sigma',pinv(Lambda));
                
                
% We save the N,F cell arrays for further use.
out_stats_file = 'stats/gpldaModel.mat'; 
disp(['Saving stats to ' out_stats_file]);
save(out_stats_file, 'gpldaModel');




