% fold - mode-k folding of a matrix X into a tensor
% -------------------------------------------------------------------------
% Syntax
%  X = fold(X, sz, K)
%
% Input
%   X    - matrix to fold
%   sz   - size of the target tensor
%   k    - mode
% 
% Output
%   X    - folded tensor
%
% See also
%  unfold
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

function X = fold(X, sz, K)

nd=length(sz);

if isnumeric(K)
  ind = {K, [K+1:nd,1:K-1]};
end

sz=sz(cell2mat(ind));
X=reshape(X,sz);
if length(ind{1})~=1 || ind{1}~=1
  [ss,indInv]=sort(cell2mat(ind));
  X=permute(X,indInv);
  % X=permute(X,[nd-jj+2:nd 1:nd-jj+1]);
end
