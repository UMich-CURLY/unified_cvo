clc
close all
clear all


rotation_angles = {'10','20','30','40','50'};
errors = {'0.1','0.2','0.3','0.4','0.5'};
rootfolder = '../../exp/';
maxNumIter = 1000;                    
% number of views
M = 4;
% cell with indexes 1:M,
idx = transpose(0:3); 
percentage = 1;
num_exp = 10;


for ai=1:length(rotation_angles)
    for ei=1:length(errors)
       
            
            currentfolder = append('toy_exp_',rotation_angles{ai},'_',errors{ei},'/');
            fprintf('start folder %s\n',currentfolder);
            error = [];
            for index = 1:num_exp
                index_s = int2str(index);
            
                % load ground truth
                try

                    gtfilename = append(rootfolder,currentfolder,index_s,'/gt_poses.txt');
                    A = readmatrix(gtfilename);
                catch
                    continue
                end
                
                Tgtr = {reshape(A(1,:),[4,4])',reshape(A(2,:),[4,4])',reshape(A(3,:),[4,4])',reshape(A(4,:),[4,4])'}';
            
                % load point cloud 
                
                
                V1 = pcread(append(rootfolder,currentfolder,index_s,'/0normal.pcd'))';
                [n,m] = size(V1);
                
                V2 = pcread(append(rootfolder,currentfolder,index_s,'/1normal.pcd'))';
                V3 = pcread(append(rootfolder,currentfolder,index_s,'/2normal.pcd'))';
                V4 = pcread(append(rootfolder,currentfolder,index_s,'/3normal.pcd'))';
                
                V1_down = pcdownsample(V1,'random',percentage,PreserveStructure=true);
                V2_down = pcdownsample(V2,'random',percentage,PreserveStructure=true);
                 V3_down = pcdownsample(V3,'random',percentage,PreserveStructure=true);
                V4_down = pcdownsample(V4,'random',percentage,PreserveStructure=true);
                V1_p = bsxfun(@plus,Tgtr{1}(1:3,1:3) *V1_down.Location',Tgtr{1}(1:3,4));
                V2_p = bsxfun(@plus,Tgtr{2}(1:3,1:3) *V2_down.Location',Tgtr{2}(1:3,4));
                V3_p = bsxfun(@plus,Tgtr{3}(1:3,1:3) *V3_down.Location',Tgtr{3}(1:3,4));
                V4_p = bsxfun(@plus,Tgtr{4}(1:3,1:3) *V4_down.Location',Tgtr{4}(1:3,4));

                for j =1:M
                    Tgtr{j} = inv(Tgtr{j});
                end

                V = {V1_p,V2_p,V3_p,V4_p}';
                % initial estimate
                % set K as the 50% of the median cardinality of the views
                K = ceil(0.5*median(cellfun(@(V) size(V,2),V))); 
                
                % sample the unit sphere, by randomly selecting azimuth / elevation angles
                az = 2*pi*rand(1,K);
                el = 2*pi*rand(1,K);
                
                %points on a unit sphere
                Xin = [cos(az).*cos(el); sin(el); sin(az).*cos(el)];% (unit) polar to cartesian conversion
                
                Xin = Xin/10; % it is good for the initialization to have initial cluster centers at the same order with the points
                % since sigma is automatically initialized based on X and V
                
                tic
                [R,t,X,S,a,~,T] = jrmpc(V,Xin,'maxNumIter',maxNumIter,'gamma',0.1);
                toc
                
                    % Convert R,t to ground truth T
                Tres = {[],[],[],[]};
                
                for j =1:M
                   Tres{j} = [[R{j},t{j}];[0,0,0,1]];
                   Tres{j} = Tres{j};
        
                end
                % measure and display convergency, view 1 is ommited as is the referential.
                fprintf('                  ||inv(A{j})*Tgtr{j} - I||_F                  \n');
                
                fprintf('______________________________________________________________\n');
                
                fprintf('Set  :'),for j=2:M,fprintf('    %d    ',j),end,fprintf('\n');
                fprintf('Error:')
                error_cur = [0];
                total_error = 0;
                for j=2:M
                    current_error = norm(Tgtr{j}-inv(Tres{1})*(Tres{j}),'fro');
                    fprintf('  %.4f ',current_error);
                    error_cur(j)=  current_error;
                    total_error = total_error + current_error;
                end
                fprintf('\n');
                error(index) = total_error;
                % log pose 
                line = zeros(4,16);
                for j = 1:M
                    line(j,:) = reshape(Tres{j}',[1,16]);
                end
                writematrix(line,append(rootfolder,currentfolder,index_s,'/jrmpc.txt'));
                writematrix(error_cur',append(rootfolder,currentfolder,index_s,'/error_jrmpc.txt'));

            end
        
        
            %log error
            writematrix(error',append(rootfolder,currentfolder,'jrmpc_error.txt'));
        
        
            
    end
end
