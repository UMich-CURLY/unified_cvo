function mat2off(C,filename)

if size(C,1)~=6 && size(C,2)~=6
    error('matrix should be of size 6xK or Kx6');
end

if size(C,1)==6
C = C'; % N x 6
end

C(:,4:6) = uint8(255*C(:,4:6));

id = fopen(filename,'w');
fwrite(id,sprintf('COFF\n%d 0 0\n',size(C,1)));

for i=1:size(C,1)
fwrite(id,sprintf('%f %f %f %d %d %d\n',C(i,:)));
end