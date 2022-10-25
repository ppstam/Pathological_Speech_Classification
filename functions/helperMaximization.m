function gmm = helperMaximization(N,F)
    N = max(N,eps);
    gmm.mu = bsxfun(@rdivide,F,N);
%     gmm.sigma = max(bsxfun(@rdivide,S,N) - gmm.mu.^2,eps);
end