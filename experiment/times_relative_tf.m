accum_tf = eye();
for i=1:size(cvof2ftracking)
    disp(i);
    A=reshape(cvof2ftracking(i,:),[4,3])';
    B=A(1:3,1:3);
    
    det1=det(B)
    ortho=B*B'
    
end