# Pathological_Speech_Classification
Pathological Speech Classification in SVD dataset using MAP Adaptation and an i-vector system.


## MAP Adaptation using data from JFA Cookbook
1. prepareSubsets.m
2. svd_jfa.m (computes trained GMMs on SVD training data adapting the jfa's ubm and outputs accurasy and confusion matrices)

## MAP Adaptation using SVD
1. prepareSubsets.m
2.    gmm_build.m (computes trained GMMs on training data)
	-edit m-file to change the number of mixtures (nmix)
3i.    score_test_utterances_two_class (outputs accuracy and confusion matrix)
	-two classes: 4 (Healthy) and 5 (Hyperfunctional Dysphonia)
	-edit m-file to change the binary variable ADAPT (1 does map adaptation; 0 does not map adaptation)
3ii.    score_test_utterances_multi_class (outputs accuracy and confusion matrix)

## i-vectors system
1. prepareSubsets.m
2. gmm_build.m (computes trained GMMs on training data)
-edit m-file to change the number of mixtures (nmix)
3. calc_baum_welch_statistics (computes zero and first order statistics for the trained GMMs)
-edit m-file to change the variable numTdim (Total variability matrix dimension)
-edit m-file to change the variable numIterations (Total variability matrix training iterations via EM algorithm)
4. pathologies_i_vectors.m (computes an i-vector for the trained GMMs)
-edit m-file to change the binary variable performLDA (1 pefrorms LDA; 0 does not perform LDA)
-edit m-file to change the binary variable performWCCN (1 pefrorms WCCN; 0 does not perform WCCN)
-edit m-file to change the variable numEigenvectors (number of eigenvectors in A matrix in LDA)
5. (optional) gplda_model.m (trains gplda model to use in test phase)
6. two_and_multi_class_testing.m (outputs accuracy and confusion matrices)
