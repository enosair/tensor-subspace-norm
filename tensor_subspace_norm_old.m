% tensor_lucky_adm - tensor denoise via lucky approach (no epsilon part)
%
% The primal problem
%  minimize 1/(2 lambda) || X - Y ||^_F + || X ||_lucky
%
% Solve the dual problem
%
%  minimize lambda/2 ||alpha||^2 - alpha' * y
%  alpha: observed entries of tensor A
%
%  s.t.  wk = Pk * inverse-observed(alpha) which is vec(unfold(A, k))
%        || wk V(k) ||_op <= 1
%
% -------------------------------------------------------------------------
% Syntax
%  [X, Z, A, fval, gap, time] = tensor_lucky_semi_denoise_adm(sz, nd, I, yy, lambda, V, varargin)
%
% Input
%   sz            - number of entries in tensor XX
%   nd            - dimension of XX
%   I             - subscript of observed entries in the unfolded matrix
%   yy            - observed entries
%   lambda        - regularization parameter
%   V             - parameter of the lucky norm
%                   V{k}: orthogomal matrix corresponding to a subspace
%
%   varargin      - list of optional varibles
%      * eta      - penalty parameter of augmented Lagrangian (default yfact/std(yy))
%      * yfact    - for eta                                   (default 10)
%      * tol      - tolerance                                 (default 1e-3)
%      * verbose  - in verbose mode                           (default 0)
%      * relative - compute relative/abosolute duality gap    (default 1)
%      * maxiter  - maximum number of iterations              (default 2000)
%
% Output
%   X             - estimated tensor X = fold(Z1,1) + ... + fold(Zk,k) + mu
%   Z             - multipliers (matrix, corresponds to components spanned
%                                by V)
%   A             - variable in the dual problem
%   fval          - objective value recorded in each iteration
%   gap           - duality gap recorded in each iteration
%   time          - run time
% -------------------------------------------------------------------------
% Change log
%
% 11/6/2014  - change the first argument from zero tensor to its size and
%              dimension. renamed variable (res -> gap). runtime recorded.
%              suppressed printing to verbose mode
%
% 01/15/2015 - drop the redundant epsilon part (now lucky is just a seminorm)
% -------------------------------------------------------------------------
% Reference
% "Estimation of low-rank tensors via convex optimization"
% Ryota Tomioka, Kohei Hayashi, and Hisashi Kashima
% arXiv:1010.0789
% http://arxiv.org/abs/1010.0789
%
% "Statistical Performance of Convex Tensor Decomposition"
% Ryota Tomioka, Taiji Suzuki, Kohei Hayashi, Hisashi Kashima
% NIPS 2011
% http://books.nips.cc/papers/files/nips24/NIPS2011_0596.pdf
%
% Convex Tensor Decomposition via Structured Schatten Norm Regularization
% Ryota Tomioka, Taiji Suzuki
% NIPS 2013
% http://papers.nips.cc/paper/4985-convex-tensor-decomposition-via-structured-schatten-norm-regularization.pdf
%
% Copyright(c) 2010-2014 Ryota Tomioka
% This software is distributed under the MIT license. See license.txt

function [X, Z, A, fval, gap, time] = tensor_subspace_norm(sz, nd, yy, lambda, V, varargin)
t0 = cputime;

opt = propertylist2struct(varargin{:});
opt = set_defaults(opt, 'eta', [], 'yfact',10, ...
    'tol', 1e-3, 'verbose', 0, 'relative',1, ...
    'maxiter',2000);

if ~isempty(opt.eta)
    eta=opt.eta;
else
    eta=opt.yfact*std(yy);
end

Z = cell(1,nd);
R = cell(1,nd);
S = cell(1,nd); % save singular values of Z{jj}V{jj}
for jj=1:nd
    szj = [sz(jj), prod(sz)/sz(jj)];
    Z{jj} = zeros(szj);
    R{jj} = zeros(szj);
end

nsv=10*ones(1,nd);   % a guess of number of sin. vals of Ztmp*V{jj} < eta
alpha = yy;
A=zeros(sz); A(:)=alpha;

fval = zeros(opt.maxiter, 1);
gap  = zeros(opt.maxiter, 1);
viol = zeros(opt.maxiter, 1);

for kk = 1:opt.maxiter
    
    % update Z
    for jj=1:nd
        Ajj = unfold(A,jj);
        Ztmp = Z{jj} + eta * Ajj;   %  z^(t-1)_k + eta * alpha^t
        [Z{jj},S{jj},nsv(jj)]=softth(Ztmp*V{jj}, eta, nsv(jj)); % S{jj} will be the singular values of Z{jj} * V{jj}
        Z{jj} = Z{jj}*V{jj}';
        R{jj} = (Ztmp-Z{jj})/eta;         % (z^(t-1)_k + eta * alpha^t-1_k - z^t_k)/eta
        viol(jj)=norm(Ajj - R{jj},'fro');
    end
    
    %     % update mu
    %     theta = mu + eta * alpha;
    %     mu_new = theta * max(1 - eta/gamma / norm(theta), 0);
    %     diff = mu - 2*mu_new ;
    %     mu = mu_new;
    
    % update alpha
    Zsum = zeros(sz);
    Rsum = zeros(sz);
    for jj=1:nd
        Zsum=Zsum+fold(Z{jj},sz,jj);
        Rsum=Rsum+fold(R{jj},sz,jj);
    end
    
    alpha = (  yy- Zsum(:) + eta * Rsum(:) ) / (lambda + eta * nd);
    %     alpha = (  yy- Zsum(:) + eta * Wsum(:) + diff + eta * alpha ) / (lambda + eta *(nd+1));
    %     if norm(beta) <= (lambda + eta * nd) / gamma
    %         alpha = beta / (lambda + eta * nd);
    %     else
    %         alpha = -beta / (norm(beta) * gamma);
    %     end
    A(:)=alpha;
    
    
    % % ---------- primal objective -----------------
    if lambda>0
        fval(kk) = 0.5*sum((Zsum(:) - yy).^2)/lambda;
        for jj = 1:nd
            fval(kk) = fval(kk) + sum(S{jj});
        end
    else
        % when lambda = 0, we should have sum_k "Zk" + mu = y. To
        % ensure that, the real components we used here is
        % Z1_k <-- Zk - errm, mu <-- mu - errm
        
        errm=(Zsum(:) - yy)/nd;
        fval(kk) = 0;
        for jj = 1:nd
            Ztmp = fold(Z{jj},sz,jj);
            Ztmp(:) = Ztmp(:) - errm;
            fval(kk) = fval(kk) + sum( svd(unfold(Ztmp,jj)*V{jj}) );
            % for the lucky norm: we didn't really compute the optimal decomposition
            % just assume the current one is so this one is an upper bound of actualy objective value
        end
    end
    
    
    % % ----------- dual certificate -----------------
    % in each iteration, pick one feasible point aa and compute the dual objective
    % didn't use aa = the alpha in the optimization directly as it violates the
    % equality constraints (unfeasible) and should lead to inf objective
    % value.
    %
    % record the best dual objective value (best certificate) in all
    % iterations
    %
    % the sequence of alpha will converge to the global optimial and so
    % does the dual certificate
    
    % the aa we pick here: scaled alpha above to satisfy: ||
    % unfold(alpha, k) V{k} ||_op <= 1, ||alpha||_F <= 1/gamma
    
    fact=1;
    for jj=1:nd
        fact=min(fact, 1/norm(unfold(A,jj)*V{jj}));
    end
%     fact = min(fact, 1/gamma / norm(alpha));
    aa = alpha*fact;
    
    
    if kk>1
        dval=max(dval, -0.5*lambda*norm(aa)^2+aa'*yy);
    else
        dval=-inf;
    end
    
    if opt.relative
        gap(kk)=1-dval/fval(kk);
    else
        gap(kk) = fval(kk) - dval;
    end
    
    if opt.verbose
        fprintf('[%d] fval=%g res=%g fact=%g viol=%s\n', kk, fval(kk), ...
            gap(kk), fact, printvec(viol));
    end
    
    % ---------- stopping criterion -----------------
    if kk>1 && gap(kk)<opt.tol % max(viol)<opt.tol && gval(kk)<opt.tol
        break;
    end
end

if opt.verbose
    fprintf('[%d] fval=%g res=%g viol=%s eta=%g\n', kk, fval(kk), ...
        gap(kk), printvec(viol),eta);
end


X=zeros(sz);
for jj=1:nd
    X = X + fold(Z{jj},sz,jj);
    %         Z{jj}=Z{jj}*nd; why muptiply by nd?
end

fval = fval(1:kk);
gap = gap(1:kk);

time = cputime - t0;
end
