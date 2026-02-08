function [XM] = fast_cal(X,M)
[~, j_indices] = max(M, [], 2);
B_sparse = sparse(1:size(M,1), j_indices, 1, size(M,1), size(M,2));
XM = X * B_sparse;
end