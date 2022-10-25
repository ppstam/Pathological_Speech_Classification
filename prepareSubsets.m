%% SVD Subsets
% Creates enrollment and test subsets for every pathology. Every subset
% contains the mfcc file paths of its speakers. The first half of the 
% pathology's data is used to create its GMM and the other half is used
% on the test phase.

clc;
clear all;

% Set up file locations
svd_dir = dir('SVD');
svd_features = dir('svd_mfcc');
svd_train_folder = 'lists/svd/train_data';
svd_test_folder = 'lists/svd/test_data';
    
% create folders to store enrollment and test sets
if ~exist(svd_train_folder, 'dir')
    mkdir(svd_train_folder);
end

if ~exist(svd_test_folder, 'dir')
    mkdir(svd_test_folder);
end

dirFlags = [svd_dir.isdir];
svd_subfolders = svd_dir(dirFlags);

% Remove . and ..
svd_subfolders(1) = [];    
svd_subfolders(1) = [];

% For each subfolder (pathology) we create two sets. The enroll set
% will be used to create a GMM representing the pathology and the 
% test set will be used in the testing phase. The two sets don't
% share common ids.
for i=1: size(svd_subfolders,1)
    
    filename = strcat('SVD/',svd_subfolders(i).name);
    files = dir([filename '/*.wav']);
    
    spk_ids = cell(1,size(files,1));
    
    % Εxtract speaker ids
    for ii=1 : size(files)  
        id = extractBefore(files(ii).name,'-');
        spk_ids{ii} = id;
    end
   
    % Remove duplicates 
    [~,ii]=unique(spk_ids,'stable');
    spk_ids=spk_ids(ii);
    
    % Every set contains half of the pathology's ids.
    % len1 -> size of the enrollment set
    % len2 -> size of the test set
    if mod(size(spk_ids,2),2)==1
        enroll_set_size = fix(size(spk_ids,2)/2);
        test_set_size = enroll_set_size+1;
    else
        enroll_set_size = (size(spk_ids,2)/2);
        test_set_size = enroll_set_size;
    end
    
    Enroll_set = cell(1,enroll_set_size);
    Test_set = cell(1,test_set_size);
    
    % Create the train and test sets containing the features' paths.
    for ii=1:enroll_set_size
        Enroll_set{ii} = ['svd_mfcc' '/' spk_ids{ii}]; 
    end
    
    for ii=1:test_set_size
        Test_set{ii} = ['svd_mfcc' '/' spk_ids{enroll_set_size + ii}]; 
    end

    % Set the paths to save the lists.
    [~,pathology,~] = fileparts(files(i).folder);
    enroll_path = [svd_train_folder '/' pathology '.lst'];
    test_path = [svd_test_folder '/' pathology '.lst'];

    % Save enrollment set
    if ~isfile(enroll_path)
        f = fopen(enroll_path, 'w');
        for j = 1:enroll_set_size
            if j==enroll_set_size
                fprintf(f, '%s', Enroll_set{j});
            else
                fprintf(f, '%s\n', Enroll_set{j});
            end
        end
        fclose(f); 
    end
    
    % Save test set
    if ~isfile(test_path)
            f = fopen(test_path, 'w');
        for j = 1:test_set_size
            if j==test_set_size
                fprintf(f, '%s', Test_set{j});
            else
                fprintf(f, '%s\n', Test_set{j});
            end
        end
        fclose(f);
    end
end