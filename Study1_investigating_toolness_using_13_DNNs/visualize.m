%% ������

clear;
s = pwd;
filepath = 'F:\TE_DCNN_RSA\DCNNs\stimuli'
cd(filepath);

net = alexnet();
net.Layers;
inputsize = net.Layers(1).InputSize;

for p = 1
    
    % ��ȡͼƬ������ͼƬ��С
    pic = imread(['SHINEd_', mat2str(p), '_300' ,'.png']);
    pic = imresize(pic,inputsize(1:2));
    pic = cat(3, pic, pic, pic);
%     
%     imgsize = size(pic);
%     imgsize = imgsize(1:2);
%     setGlobalimgsize(imgsize);
    
%     analyzeNetwork(net);
    
    % ��ȡÿһ�������
%     act5 = activations(net, pic, 'pool1');
%     act9 = activations(net, pic, 'pool2');
%     act16 = activations(net, pic, 'pool5');
%     act17 = activations(net, pic, 'fc6');
%     act20 = activations(net, pic, 'fc7');
%     act23 = activations(net, pic, 'fc8');
    
    % ����
%     top5classify(net, pic);
    
    % �鿴ĳһ��ȫ��
%     showact(act16);
    
    % �鿴ĳһͨ��
%     specificact(pic, act5, 1);
    
    % �鿴��󼤻������
%     [maxvalue, row, col, depth] = maxact(pic, act20);
    
    % deepdream����
    layer = 'fc8';
    channels = 418:419;
    classes = net.Layers(end).Classes(channels);
    paramidlevels = 1;
    iteration = 100;
    
    deepdreamgenerator(net, pic, layer, channels, ...
    paramidlevels, 0, iteration);
    
end

cd(s);


%% ����

function setGlobalimgsize(val)
    global IMGSIZE
    IMGSIZE = val;
end

function x = getGlobalimgsize
    global IMGSIZE
    x = IMGSIZE;
end

% top5����
function top5classify(net, pic)
    [label,scores] = classify(net,pic);
    [~,idx] = sort(scores,'descend');
    idx = idx(5:-1:1);
    classNamesTop = net.Layers(end).ClassNames(idx);
    scoresTop = scores(idx);
    figure
    barh(scoresTop)
    xlim([0 1])
    title('Top 5 Predictions')
    xlabel('Probability')
    yticklabels(classNamesTop)
end

% �ò���ĳһ���ȫ��չʾ
function showact(act)
    sz = size(act);
    act = reshape(act,[sz(1) sz(2) 1 sz(3)]);
    I = imtile(mat2gray(act),'GridSize',[10 10]);
    imshow(I);
end

% �ò���ĳ��ĳһ��ͨ����ԭͼ�ĶԱ�չʾ
function specificact(pic, act, channel)
    act1ch = act(:,:,channel);
    act1ch = mat2gray(act1ch);
    IMGSIZE = getGlobalimgsize;
    act1ch = imresize(act1ch,IMGSIZE);
    I = imtile({pic,act1ch});
    imshow(I);
end

% �ò����ļ���ͨ��
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

% deepdream
function I = deepdreamgenerator(net, initialimage, layer, channels, ...
    paramidlevels, verbose, iteration)
    I = deepDreamImage(net,layer,channels,...
        'PyramidLevels',paramidlevels,'Verbose',verbose,...
        'InitialImage', initialimage, 'NumIterations', iteration);
    figure
    I = imtile(I, 'ThumbnailSize', [250 250]);
    imshow(I);
end