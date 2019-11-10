path="/home/justin/research/outdoor_cvo/";
gt_file="/media/justin/LaCie/data/kitti/sequences/05/groundtruth.txt";
in_prod_file="inner_product_all.txt";
cvo_result_file="cvo_f2f_tracking.txt";

gt_id = fopen(gt_file);
in_prod_id = fopen(path+in_prod_file);
cvo_result_id = fopen(path+cvo_result_file);


gt = fscanf(gt_id,'%f \n');
in_prod_all = fscanf(in_prod_id,'%f \n');
cvo_result = fscanf(cvo_result_id,'%f \n');

gt = reshape(gt,12,[])';
in_prod_all = reshape(in_prod_all,3,[])';
cvo_result = reshape(cvo_result,12,[])';
% remove first identity in gt
gt(1,:)=[];

num_data=size(cvo_result,1);
num_nonzero_in_A = zeros(num_data,1);
in_prod = zeros(num_data,1);
in_prod_normalized = zeros(num_data,1);
trans_error = zeros(num_data,1);
log_error = zeros(num_data,1);
omega_error = zeros(num_data,1);
v_error = zeros(num_data,1);
for i=1:num_data
    
    if i==1
        gt_i = reshape(gt(i,:),4,3)';
        gt_i = [gt_i; 0 0 0 1];
        
        cvo_i = reshape(cvo_result(i,:),4,3)';
        cvo_i = [cvo_i; 0 0 0 1];
    else
        gt_im1_0 = reshape(gt(i-1,:),4,3)';
        gt_im1_0 = [gt_im1_0; 0 0 0 1];
        gt_i_0 = reshape(gt(i,:),4,3)';
        gt_i_0 = [gt_i_0; 0 0 0 1];
        gt_i = inv(gt_im1_0)*gt_i_0;
        
        cvo_im1_0 = reshape(cvo_result(i-1,:),4,3)';
        cvo_im1_0 = [cvo_im1_0; 0 0 0 1];
        cvo_i_0 = reshape(cvo_result(i,:),4,3)';
        cvo_i_0 = [cvo_i_0; 0 0 0 1];
        cvo_i = inv(cvo_im1_0)*cvo_i_0;
    end
    
    % calculate translational error
    dx = cvo_i(1,4)-gt_i(1,4);
    dy = cvo_i(2,4)-gt_i(2,4);
    dz = cvo_i(3,4)-gt_i(3,4);
    trans_error(i)= sqrt(dx^2+dy^2+dz^2);
    
%     log_error(i) = sum(sum(logm(gt_i\cvo_i)));
    lie_err = logm(gt_i\cvo_i);
    omega = lie_err(3,1:3);
    v = lie_err(3,1:3);
    
    log_error(i)=norm(omega)+norm(v);
    omega_error(i) = norm(omega);
    v_error(i) = norm(v);
    
    num_nonzero_in_A(i)=in_prod_all(i,1);
    in_prod(i) = in_prod_all(i,2);
    in_prod_normalized(i) = in_prod_all(i,3);
    
end

p1 = figure(1);
scatter(trans_error,num_nonzero_in_A,'.');
xlabel("translational error");
ylabel("number of non zeros in A");
title("number of non zeros in A v.s. translational error");

p2 = figure(2);
scatter(trans_error,in_prod,'.');
xlabel("translational error");
ylabel("inner product");
title("inner product v.s. translational error");

p3 = figure(3);
scatter(trans_error,in_prod_normalized,'.');
xlabel("translational error");
ylabel("normalized inner product");
title("normalized inner product v.s. translational error");

p4 = figure(4);
scatter(log_error,num_nonzero_in_A,'.');
xlabel("logarithmic error");
ylabel("number of non zeros in A");
title("number of non zeros in A v.s. logarithmic error");

p5 = figure(5);
scatter(log_error,in_prod,'.');
xlabel("logarithmic error");
ylabel("inner product");
title("inner product v.s. logarithmic error");

p6 = figure(6);
scatter(log_error,in_prod_normalized,'.');
xlabel("logarithmic error");
ylabel("normalized inner product");
title("normalized inner product v.s. logarithmic error");

p7 = figure(7);
scatter(omega_error,num_nonzero_in_A,'.');
xlabel("omega error");
ylabel("number of non zeros in A");
title("number of non zeros in A v.s. omega error");

p8 = figure(8);
scatter(omega_error,in_prod,'.');
xlabel("omega error");
ylabel("inner product");
title("inner product v.s. omega error");

p9 = figure(9);
scatter(omega_error,in_prod_normalized,'.');
xlabel("omega error");
ylabel("normalized inner product");
title("normalized inner product v.s. omega error");

p10 = figure(10);
scatter(v_error,num_nonzero_in_A,'.');
xlabel("v error");
ylabel("number of non zeros in A");
title("number of non zeros in A v.s. v error");

p11 = figure(11);
scatter(v_error,in_prod,'.');
xlabel("v error");
ylabel("inner product");
title("inner product v.s. v error");

p12 = figure(12);
scatter(v_error,in_prod_normalized,'.');
xlabel("v error");
ylabel("normalized inner product");
title("normalized inner product v.s. v error");