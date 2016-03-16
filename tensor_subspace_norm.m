function [X, Z, A, fval, gap, time] = tensor_subspace_norm(sz, nd, yy, lambda, V, varargin)
%
%  Minimize the tensor subspace norm by solving the scaled dual problem
%
%  min            lambda/2 ||A||^2 - A' * y
%  A, {W(k)}
%  s.t.           || W(k) ||_op <= 1
%                 W(k) = D(k) V(k)
% -------------------------------------------------------------------------
%
% Input
%   sz            - size of target tensor X
%   nd            - dimension of X
%   yy            - noisy observation
%   lambda        - regularization parameter
%   V             - associated subspace
%                   V{k}: orthogomal matrix corresponding to a subspace
%   varargin      - list of optional varibles
%      * eta      - penalty parameter of augmented Lagrangian (default 10/std(yy))
%      * tol      - tolerance                                 (default 1e-3)
%      * verbose  - print output                              (default 0)
%      * relative - compute relative/abosolute duality gap    (default 1)
%      * maxiter  - maximum number of iterations              (default 2000)
%
% Output
%   X             - estimated tensor X
%   Z             - components spanned by {V(k)} : X = sum_k Z(k)
%   A             - dual tensor
%   fval          - objective value recorded in each iteration
%   gap           - duality gap recorded in each iteration
%   time          - run time
%
% -------------------------------------------------------------------------
% Reference
%   Interpolating Convex and Non-Convex Tensor Decompositions via the Subspace Norm
%   Qinqing Zheng, Ryota Tomioka
%   NIPS 2015
%   
% Copyright(c) 2010-2016 Ryota Tomioka, Qinqing Zheng
% This software is distributed under the MIT license. See license.txt

t0 = cputime;

opt = propertylist2struct(varargin{:});
opt = set_defaults(opt, 'eta', [], 'tol', 1e-3, 'verbose', 0, 'relative',1, 'maxiter',2000);

if ~isempty(opt.eta)
    eta = opt.eta;
else
    eta = 10*std(yy);
end


Z = cell(1, nd); % components of X
M = cell(1, nd); % multiplier of dual problem
S = cell(1, nd); % save singular values of M{jj}
R = cell(1, nd); 
T = cell(1, nd);
for jj=1:nd
    szj = [sz(jj), prod(sz)/sz(jj)];
    M{jj} = zeros([sz(jj), size(V{jj}, 2)]);
    Z{jj} = zeros(szj);
    R{jj} = zeros(szj);
    T{jj} = zeros(szj);
end

nsv = 10*ones(1,nd);   % a guess of number of sin. vals that < eta
alpha = yy;
A = zeros(sz); A(:)=alpha;

fval = zeros(opt.maxiter, 1);
gap  = zeros(opt.maxiter, 1);
viol = zeros(opt.maxiter, 1);

for kk = 1:opt.maxiter
    
    % update M & Z
    for jj=1:nd
        Ajj = unfold(A,jj);
        Mjj_old = M{jj};
        Mtmp = M{jj} + eta * Ajj * V{jj} ;  
        [M{jj},S{jj},nsv(jj)] = softth(Mtmp, eta, nsv(jj)); % S{jj} will be the singular values of M{jj}
        Z{jj} = M{jj} * V{jj}';
        R{jj} = (Mtmp - M{jj})/eta;         
        viol(jj) = norm(Ajj * V{jj} - R{jj},'fro');
        T{jj} = (2 * M{jj} - Mjj_old ) * V{jj}';
    end
    
    
    % update dual tensor 
    Zsum = zeros(sz);
    Tsum = zeros(sz);
    for jj=1:nd
        Zsum = Zsum + fold(Z{jj},sz,jj);
        Tsum = Tsum + fold(T{jj} ,sz,jj);
    end
    
    alpha = (  yy + nd * eta * alpha - Tsum(:) ) / (lambda + eta * nd);
    A(:)=alpha;
    
    
    % % ---------- primal objective -----------------
    if lambda > 0
        fval(kk) = 0.5*sum((Zsum(:) - yy).^2)/lambda;
        for jj = 1:nd
            fval(kk) = fval(kk) + sum(S{jj});
        end
    else
        % when lambda = 0, we should have sum_k "Zk" + mu = y. To
        % ensure that, the real components we used here is
        % Z1_k <-- Zk - errm, mu <-- mu - errm
        %
        % we didn't really compute the optimal decomposition
        % this gives an upper bound of actual objective value
        
        errm=(Zsum(:) - yy)/nd;
        fval(kk) = 0;
        for jj = 1:nd
            Ztmp = fold(Z{jj},sz,jj);
            Ztmp(:) = Ztmp(:) - errm;
            fval(kk) = fval(kk) + sum( svd( unfold(Ztmp, jj) * V{jj}) );
        end
    end
    
    
    % % ----------- dual certificate -----------------
    % in each iteration, pick one feasible point aa and compute the dual objective
    % record the best best certificate in all iterations
    %
    % the sequence of alpha will converge to the global optimial and so
    % does the dual certificate
    %
    % didn't use aa = the alpha in the optimization directly as it violates the
    % equality constraints (unfeasible) and should lead to inf objective value.
    %
    % the aa we pick here: scaled alpha above to satisfy: 
    % || unfold(alpha, k) V{k} ||_op <= 1
    
    fact=1;
    for jj=1:nd
        fact=min(fact, 1/norm(unfold(A,jj)*V{jj}));
    end
    aa = alpha * fact;
    
    
    if kk>1
        dval=max(dval, -0.5*lambda*norm(aa)^2+aa'*yy);
    else
        dval=-inf;
    end
    
    if opt.relative
        gap(kk) = 1 - dval/fval(kk);
    else
        gap(kk) = fval(kk) - dval;
    end
    
    if opt.verbose
        fprintf('[iter %d] fval=%g gap=%g fact=%g viol=%s\n', kk, fval(kk), gap(kk), fact, printvec(viol));
    end
    
    % ---------- stopping criterion -----------------
    if kk>1 && gap(kk) < opt.tol
        break;
    end
end


X=zeros(sz);
for jj=1:nd
    X = X + fold(Z{jj},sz,jj);
end

fval = fval(1:kk);
gap = gap(1:kk);

time = cputime - t0;
end
