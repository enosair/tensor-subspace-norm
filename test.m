clear
rng(941);

%%% construct data
n1 = 20; n2 = 30; n3 =40; 
sz = [n1,n2,n3]; nd = length(sz);
ntotal = prod(sz);
r = 2; % rank

[U,~,~] = svd(randn(n1));
U = U(:, 1:r);

[V,~,~] = svd(randn(n2));
V = V(:, 1:r);

[W,~,~] = svd(randn(n3));
W = W(:, 1:r);
comp1 = kolda3(1, U(:,1), V(:,1), W(:,1));
comp2 = kolda3(1, U(:,2), V(:,2), W(:,2));

beta1 = 10; beta2 = 20;

X = comp1 * beta1 + comp2 * beta2 ; % ground truth
sigma = beta1 / ntotal^0.25 * 0.1;  % noise std
Y = X + sigma * randn(sz);          % observation

%%% subspace norm minimization

ind = (1:prod(sz))';
[I,J,K] = ind2sub(sz, ind);
tol = 1e-3;

rr = 2; % input rank
U = cell(nd, 1); V = cell(nd, 1);
for i = 1:nd
    [uu, ss] = eig(unfold(Y, i) * unfold(Y, i)' );
    if (ss(1) > ss(end))
        U{i} = uu(:, 1:rr);
    else
        U{i} = uu(:, end-(rr-1):end);
    end
end
V{1} = kron(U{3}, U{2});
V{2} = kron(U{1}, U{3});
V{3} = kron(U{2}, U{1});


lambda = 1;

Xhat = tensor_subspace_norm(sz, nd, Y(:), lambda, V, 'tol', tol, 'verbose', 1, 'maxiter', 1000);
err = norm(Xhat(:)- X(:)) / norm(X(:));
fprintf('subspace norm :: sigma = %.3e  lambda = %.3e relative error = %f\n', sigma, lambda, err);