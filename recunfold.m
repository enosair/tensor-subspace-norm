function v = recunfold(X)
% Input
%   X   - n x n x n symmetric tensor
%
% Output
%   v   - principal eigenvector obtained by recursive unfolding

n = size(X,1);

[~, ~, V] = svds(unfold(X, 1), 1);

[U, ~, ~] = svd(reshape(V(:,1), n, n));

v = U(:,1);
end
