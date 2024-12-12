
clear;
s = pwd;
cd('F:\TE_DCNN_RSA\DCNNs\stimuli'); % this is where the stimuli images locate

% loading model
net = alexnet;
net.Layers;
inputSize = net.Layers(1).InputSize;

for p = 1%:80
    
    % input and resize
    pic = imread(['SHINEd_', mat2str(p), '_300' ,'.png']);
    pic = imresize(pic,inputSize(1:2));
    pic = cat(3, pic, pic, pic);
    % pic = mat2gray(rgb);
    % imshow(pic)
    imgsize = size(pic);
    imgsize = imgsize(1:2);
    setGlobalimgsize(imgsize);

    % get features in different layers
    act5 = activations(net, pic, 'pool1');
    act9 = activations(net, pic, 'pool2');
    act16 = activations(net, pic, 'pool5');
    act17 = activations(net, pic, 'fc6');
    act20 = activations(net, pic, 'fc7');
    act23 = activations(net, pic, 'fc8');

    %% only a try, this part is not included in my program
    % 查看某一层全景
%     showact(act5);
    % 查看某一通道
%     specificact(pic, act5, 1);
    % 查看最大激活及其索引
%     [maxvalue, row, col, depth] = maxact(pic, act16);
    
    layer{p, 1} = act5;
    layer{p, 2} = act9;
    layer{p, 3} = act16;
    layer{p, 4} = act17;
    layer{p, 5} = act20;
    layer{p, 6} = act23;
end

cd(s);
% % save
% filename=strcat('AlexnetLayersOutput.mat');
% save (filename, 'layer')

function setGlobalimgsize(val)
    global IMGSIZE
    IMGSIZE = val;
end

function x = getGlobalimgsize
    global IMGSIZE
    x = IMGSIZE;
end

% 该层在某一层的全景展示
function showact(act)
    sz = size(act);
    act = reshape(act,[sz(1) sz(2) 1 sz(3)]);
    I = imtile(mat2gray(act),'GridSize',[10 10]);
    imshow(I);
end

% 该层在某层某一个通道和原图的对比展示
function specificact(pic, act, channel)
    act1ch = act(:,:,channel);
    act1ch = mat2gray(act1ch);
    IMGSIZE = getGlobalimgsize;
    act1ch = imresize(act1ch,IMGSIZE);
    I = imtile({pic,act1ch});
    imshow(I);
end

% 该层最大的激活通道
function [maxValue, maxRow, maxCol, maxDepth] = maxact(pic, act)
    [maxValue,maxValueIndex] = max(act(:));
    [maxRow, maxCol, maxDepth] = ind2sub(size(act), maxValueIndex);
    [~,maxValueIndex] = max(max(max(act)));
    actMax = act(:,:,maxValueIndex);
    actMax = mat2gray(actMax);
    IMGSIZE = getGlobalimgsize;
    actMax = imresize(actMax,IMGSIZE);
    I = imtile({pic,actMax});
    imshow(I);
end
