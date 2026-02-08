function [M,y] = PSIG(X)
V=length(X);

for i = 1 :length(X)
    [idx,~, center] = signal_FINCH(X{i}',[]);
    Q{i}=full(ind2vec(idx')');
    adj{i} = sparse(double(idx == idx'));
end

adj_sum = adj{1};
for i = 2:length(adj)
    adj_sum = adj_sum + adj{i};
end

T = adj_sum >= fix(V / 2) + 1;
[~, y] = graphconncomp(T, 'Directed', false);
M=ind2vec(y)';

% M=full(M);
end