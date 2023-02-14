% DEMOJRMPCSYNTHETIC   Example of using jrmpc into synthetic data.
%    This example loads numel(theta) views from ./syntheticData/ and calls
%    jrmpc to do the registration. It creates 4 plots one with the initial
%    position of the point sets, one which shows the registration at
%    every iteration, one with the final alignment achieved after maxNumIter
%    iterations and one with the "cleaned up" point sets. Directory 
%    ./syntheticData/ contains 4 partial views from the stanford bunny, 
%    each view is degraded with disparsity noise and outliers. The angles in
%    theta are ground truth angles (same for all 3 axes) used in the 
%    construction.
%
%    $ 18 / 12 / 2014 3:24 PM $

clc
close all
clear all

% given an angle theta in radians angle2rotation(theta) construtcts a 
% rotation matrix, rotating a vector with angle theta along all three axes.
% Right hand convention is used for the rotation around the perpendicular.
angle2rotation = @(theta) [1 0 0;0 cos(theta) -sin(theta);0 sin(theta) cos(theta)] ...
                         *[cos(theta) 0 sin(theta);0 1 0;-sin(theta) 0 cos(theta)] ...
                         *[cos(theta) -sin(theta) 0;sin(theta) cos(theta) 0; 0 0 1];
 
% number of iterations to be run
maxNumIter = 100;                    
                     
% latent angles, rotating V{j} by theta(j) reprojects it to V{1} rotated by
% theta(1). used to quantify the accuracy of the estimated R
theta = [0; pi/20; pi/10; pi/6];

% number of views, M files must be found in the directory ./syntheticData 
M = numel(theta);

% cell with indexes 1:M, used as suffixes on view's filenames
idx = transpose(1:M);

% string-labels for legends in subsequent plots
strIdx = arrayfun(@(j) sprintf('view %d',j),idx,'uniformoutput',false);

fprintf(' Data loading...\n');

% load the views, file view<j>.txt corresponds to theta(j)
% cutView*.txt is a partial view as described in the paper, while view*.txt
% "sees" the whole surface (again downsampled and noisy)

%V = arrayfun(@(j) dlmread(sprintf('./syntheticData/view%d.txt',j),' ')',idx,'uniformoutput',false);
V = arrayfun(@(j) dlmread(sprintf('./syntheticData/cutView%d.txt',j),' ')',idx,'uniformoutput',false);


% ground truth rotation matrices Rgt{j}*V{j} is aligned with Rgt{1}*R{1}
Rgt = arrayfun(@(theta) angle2rotation(theta),theta,'uniformoutput',false);

% colors for each view
clrmap = {[1 .1412 0]; [.1373 .4196 .5569]; [0 0 1]; [.8039 .6078 .1137]};

% markerSizes 
markerSize = {7; 70; 12; 10};

% markers
marker = {'s'; 'x'; '.'; '^'};

% initialize GMM means Xin, using random sampling of a unit sphere. Choose
% your own initialization. You may want to initialize Xin with some of the
% sets.

% set K as the 50% of the median cardinality of the views
K = ceil(0.5*median(cellfun(@(V) size(V,2),V))); 

% sample the unit sphere, by randomly selecting azimuth / elevation angles
az = 2*pi*rand(1,K);
el = 2*pi*rand(1,K);

%points on a unit sphere
Xin = [cos(az).*cos(el); sin(el); sin(az).*cos(el)];% (unit) polar to cartesian conversion

Xin = Xin/10; % it is good for the initialization to have initial cluster centers at the same order with the points
% since sigma is automatically initialized based on X and V

% show the initial position of the point clouds
figure(1);
hold on,grid on

% make the legend
title('Initial position of the point clouds','fontweight','bold','fontsize',12);

hg1 = cellfun(@(V,clrmap,marker,markerSize)  scatter3(V(1,:),V(2,:),V(3,:),markerSize,clrmap,marker),V,clrmap,marker,markerSize);
size(hg1)
legend(strIdx{:});

set(1,'position',get(1,'position')+[-260 0 0 0]);

set(gca,'fontweight','bold','children',hg1);

view([40 54]) 
scatter3(Xin(1,:),Xin(2,:),Xin(3,:),'k')
hold off; drawnow

fprintf('Data registration... \n\n');

% call JRMPC (type jrmpc with no arguments to see the documentation).
[R,t,X,S,a,~,T] = jrmpc(V,Xin,'maxNumIter',maxNumIter,'gamma',0.1);

% measure and display convergency, view 1 is ommited as is the referential.
fprintf('                  ||Rgt{j} - R{j}^T*R{1}||_F                  \n');

fprintf('______________________________________________________________\n');

fprintf('Set  :'),for j=2:M,fprintf('    %d    ',j),end,fprintf('\n');
fprintf('Error:'),for j=2:M,fprintf('  %.4f ',norm(Rgt{j}-R{j}'*R{1},'fro'));end

fprintf('\n');


% visualize the registration process, see documentation of jrmpc for T.

figure(2);

for iter = 1:maxNumIter
    % apply transformation of iteration : iter
    TV = cellfun(@(V,R_iter,t_iter) bsxfun(@plus,R_iter*V,t_iter),V,T(:,1,iter),T(:,2,iter),'uniformoutput',false);
    
    clf(2);
    
    hold on, grid on
    
    title(sprintf('Registration of the sets after %d iteration(s).\n',iter),'fontweight','bold','fontsize',12);
    
    hg2 = cellfun(@(TV,clrmap,marker,markerSize) scatter3(TV(1,:),TV(2,:),TV(3,:),markerSize,clrmap,marker), TV, clrmap, marker, markerSize);
    
    legend(strIdx{:});
    
    set(2,'position',get(1,'position')+[+580 0 0 0]);
    
    % iteration 1 locks the axes of subsequent plots
    if iter == 1
       XLim = get(gca,'XLim');
       
       YLim = get(gca,'YLim');
       
       Zlim = get(gca,'ZLim');
       
       set(gca,'fontweight','bold','children',hg2);
    else
       set(gca,'XLim',XLim,'YLim',YLim,'ZLim',Zlim,'fontweight','bold','children',hg2); 
    end

    view([40 54])
    
    hold off
    
    pause(.12);
end

% detect and remove "bad" centers and "unreliable" points 
[TVrefined,Xrefined,Xrem] = removePointsAndCenters(TV,X,S,a);
    
% visualize TVrefined.
figure(3);
hold on, grid on

    title('Final registration with unreliable points removed','fontweight','bold','fontsize',12);
    
    hg3 = cellfun(@(TVrefined,clrmap,marker,mkSize) scatter3(TVrefined(1,:),TVrefined(2,:),TVrefined(3,:),mkSize,clrmap,marker),TVrefined,clrmap,marker,markerSize);
    
    legend(strIdx{:});
    
    % use the same axes as in the registration process
    set(gca,'XLim',XLim,'YLim',YLim,'ZLim',Zlim,'fontweight','bold','children',hg3);
    
    set(3,'position',get(1,'position')+[0 -510 0 0]);
    
    view([40 54]) 
hold off

% Visualize bad centers (orange) and good centers (blue).
figure(4);
hold on, grid on

    title('Final GMM means.','fontweight','bold','fontsize',12);
    
    scatter3(Xrefined(1,:),Xrefined(2,:),Xrefined(3,:),8,[0 .38 .67],'s');
    
    scatter3(Xrem(1,:),Xrem(2,:),Xrem(3,:),40,[1 .1412 0],'marker','x');
    
    legend('"Good" Centers','"Bad" Centers');
    
    % use the same axes as in the registration process
    set(gca,'XLim',XLim,'YLim',YLim,'ZLim',Zlim,'fontweight','bold');
    
    set(4,'position',get(1,'position')+[+580 -510 0 0]);
    
    view([40 54])
    
hold off
