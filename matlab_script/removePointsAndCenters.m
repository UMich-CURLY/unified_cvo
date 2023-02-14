%REMOVEPOINTSANDCENTERS refines THE point sets in TV using S and a.
%   TVrefined = removePointsAndBadCenters(TV,X,S,a)
%   It diregards components with high variance, and then
%   it applies MAP from TV to X plus outlier class using posteriors 'a' and 
%   it finally returns in TVrefined only points classified to low-variance
%   components.
%   
%   [TVrefined,Xrefined] = removePointsAndCenters(TV,X,S,a)
%   Returns also the "good" GMM means in Xrefined.
%
%   [TVrefined,Xrefined,Xrem] = removePointsAndCenters(TV,X,S,a)
%   Return also the remaining "bad" centers from X in Xrem.
%
%   [TVrefined,Xrefined,Xrem,Irefined] = removePointsAndCenters(TV,X,S,a,I)
%   The 5th input argument I is a colormap for TV. The function expects as I
%   a cell array, same size as TV, with elements [numel(TV{j}) x 3] matrices
%   with normalized RGB color triplets as rows. Irefined has the same size as
%   TVrefined and contains the color for the points in TVrefined.
%
%   $ 7 / 8 / 2014 2:24 PM $
function [TVrefined,Xrefined,X,Irefined] = removePointsAndCenters(TV,X,S,a,I)

% compare variances to extract centers with dense distributions, i.e inliers
dense_k = find(S < 2.5*median(S));
%keyboard
% classify points in V to centers 1,..,K or the outiler class K+1
[~,classifyV] = cellfun(@(a) max([a 1-sum(a,2)],[],2),a,'uniformoutput',false);

% for each view, extract indexes of all points classified into low-variance clusters
% >> find() may return multiple indexes for a specific view and k, therefore
%    concatenation ability of the output of arrayfun cannot be taken for granted.
reliablePointsOnV = cellfun(@(classifyV) arrayfun(@(k) find(classifyV == k),dense_k,'uniformoutput',false), classifyV, 'uniformoutput',false);

% multiple assignments for a center are into cells to concatenate all indexes
reliablePointsOnV = cellfun(@(reliablePointsOnVj) cat(1,reliablePointsOnVj{:}), reliablePointsOnV, 'uniformoutput',false);

% extract "reliablePoints" from V, or TV to not transform V again
TVrefined = cellfun(@(TV,reliablePointsOnV) TV(:,reliablePointsOnV),TV,reliablePointsOnV,'uniformoutput',false);

% discard centers of "flat" components
Xrefined = X(:,dense_k);

% if Xrem is requested compute by [] X, return X as Xrem
if nargout > 2
    X(:,dense_k) = [];
end

% If color information is provided, remove colorInfo for outlier points
if nargin > 4
    Irefined = cellfun(@(I,indexes) I(indexes,:),I,reliablePointsOnV,'uniformoutput',false);
end
