% DEMOJRMPCTOF   Example of using JRMPC in the registration of TOF data.
%    This demo loads M views from ./tofData/ and calls jrmpc to do the reg-
%    istration, then visualizes the result in 2 figures, one depicting the
%    final registration with all the points and another showing the reg-
%    istration with "unreliable points" removed. The color
%    value for each point was acquired by a stereo pair rigidly attached 
%    to the TOF camera. The colors are only needed for
%    the visualization and are not used by the algorithm at any step.
%
%    $ 18 / 12 / 2014 11:00 AM $

clc
close all
clear all

% indices of views to be read from ./tofData 
% filenames: view1.txt, view2.txt ...

M = 10;%the number of views, Nj is the cardinality of view V{j}
idx = transpose(1:M);


% construct file names to read the view*.txt files and make the legends
fname = arrayfun(@(idx) sprintf('./tofData/view%d.txt',idx),idx,'uniformoutput',false);

fprintf('TOF data loading from ''./tofData/''.\n');
% .txt files are organised in 6 columns, [x y z R G B], with the leftmost 3
% containing floating point indices and the rightmost 3 containing the color
% of the points as unsigned integers in the range [0,255], as in 24-bit RGB.
V = cellfun(@(fname) dlmread(fname,' '),fname,'uniformoutput',false);

% V now contains 6 x Nj matrices with coordinates and colors, separated 
% into point coordinates (V) and color info normalized in [0,1] (I).
% I is converted into a M x 1 cell array with 1 x Nj cell arrays with
% 3 x 1 vector-colors each, so we can feed each plot3 with a different color

 
% %%% optionally work with less points, donwnsample the point-sets by a factor df >1
df=1; % df=1 means no downsampling
[V,I] = cellfun(@(V) deal(V(1:df:end,1:3)',double(V(1:df:end,4:6))/255),V,'uniformoutput',false);


% initialize centers, the median cardinality can also be used as K but 
% it will dramatically increase the computational complexity without 
%  substantial gain in performance. Higher K can be combined with point-set
%  downsampling to avoid high complexity
K = 450;

% sample the unit sphere, by randomly selecting azimuth / elevation angles.
az = 2*pi*rand(1,K);
el = 2*pi*rand(1,K);

% (unit) polar to cartesian conversion.
Xin = [cos(az).*cos(el); sin(el); sin(az).*cos(el)];

Xin = Xin*100; % make them have the same order with points (it helps the convergence)


% choose the middle pointset instead to initialize the cluster centers
% Xin = V{5}(:,unique(round(linspace(1,size(V{5},2),K))));


% Number of Iterations.
maxNumIter = 100;
 

% call jrmpc to do the actual compuation, 
fprintf('Joint registration...(it takes a few minutes with full sets).\n');
[R,t,X,S,a] = jrmpc(V,Xin,'maxNumIter',maxNumIter,'gamma',0.1, 'epsilon', 1e-5);

% apply transformation to each view V{j} as R{j}*V{j}+t{j}*ones(1,Nj). Then
% segment each TV{j} into an 1 x Nj cell array with 3 x 1 vectors. This allows
% calling plot3 with the corresponding color for each point
TV = cellfun(@(V,R,t) bsxfun(@plus,R*V,t),V,R,t,'uniformoutput',false);

% visualize the final result of the registration.
fprintf('ploting...(it takes some time with full sets).\n');

figure(1)
hold on
title(sprintf('Final registration of %d TOF images',M),'fontweight','bold','fontsize',12)

%scatter3(TV{1}(1,:),TV{1}(2,:),TV{1}(3,:),7,I{1},'filled')

cellfun(@(TV,I) scatter3(TV(1,:),TV(2,:),TV(3,:),7,I,'filled'),TV(1:M),I(1:M));

set(1,'position',get(1,'position')+[-260 0 0 0]);

view([0 -70])
hold off

% use S and a to detect and remove unreliable points
[TVrefined,~,~,Irefined] = removePointsAndCenters(TV,X,S,a,I);

% visualize with color the finals with outliers removed
figure(2)
hold on
title('Registration after removing points classified on clusters with high variance','fontweight','bold','fontsize',12)

cellfun(@(TVref,Iref) scatter3(TVref(1,:),TVref(2,:),TVref(3,:),7,Iref,'filled'),TVrefined,Irefined);

set(2,'position',get(1,'position')+[+580 0 0 0]);

view([0 -70])
hold off

