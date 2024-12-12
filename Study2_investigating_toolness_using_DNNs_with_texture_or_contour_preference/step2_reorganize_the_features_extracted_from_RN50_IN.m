clear;

import npy_matlab.*

s = pwd;

for p = 1:80
    path = 'F:\TE_DCNN_RSA\DCNNs\Resnet-50\IN_2\maxpool';
    cd(path);
    filename = ['maxpool_', mat2str(p), '_feats.npy'];
    layer{p, 1} = readNPY(filename);
end

for p = 1:80
    path = 'F:\TE_DCNN_RSA\DCNNs\Resnet-50\IN_2\layer1';
    cd(path);
    filename = ['layer1_', mat2str(p), '_feats.npy'];
    layer{p, 2} = readNPY(filename);
end

for p = 1:80
    path = 'F:\TE_DCNN_RSA\DCNNs\Resnet-50\IN_2\layer2';
    cd(path);
    filename = ['layer2_', mat2str(p), '_feats.npy'];
    layer{p, 3} = readNPY(filename);
end

for p = 1:80
    path = 'F:\TE_DCNN_RSA\DCNNs\Resnet-50\IN_2\layer3';
    cd(path);
    filename = ['layer3_', mat2str(p), '_feats.npy'];
    layer{p, 4} = readNPY(filename);
end

for p = 1:80
    path = 'F:\TE_DCNN_RSA\DCNNs\Resnet-50\IN_2\avgpool';
    cd(path);
    filename = ['avgpool_', mat2str(p), '_feats.npy'];
    layer{p, 5} = readNPY(filename);
end

for p = 1:80
    path = 'F:\TE_DCNN_RSA\DCNNs\Resnet-50\IN_2\fc';
    cd(path);
    filename = ['fc_', mat2str(p), '_feats.npy'];
    layer{p, 6} = readNPY(filename);
end

cd(s);
% save
filename=strcat('resnet50LayersOutput.mat');
save (filename, 'layer')
