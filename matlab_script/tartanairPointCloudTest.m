folderpath = '../demo_data/result_cvo';
frames = [0,1,3,5];
for i= 1:4
    filepath = append(folderpath,int2str(frames(i)),'.pcd')
    pt = pcread(filepath);
    pcshow(pt);
end
