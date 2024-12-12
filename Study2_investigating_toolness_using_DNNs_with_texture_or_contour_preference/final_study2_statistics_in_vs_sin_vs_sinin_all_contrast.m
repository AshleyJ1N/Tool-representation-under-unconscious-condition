clear;
addpath(pwd); 

%% parameter
type = "cfs";  % cfs or visible
alpha = 0.05;
sampling = 0.005;
layer = 5;
correct_for_model = 'cluster';
correct_for_layers = 'cluster';
correct_for_layer_vs_0 = 'cluster';
sided_for_model = 0; % 0 or 1 or -1
sided_for_layers = 0; % 0 or 1 or -1
sided_for_layer_vs_0 = 0; % 0 or 1 or -1
seed = 20241021;
if type == "cfs"
    sub = 21;
elseif type == "visible"
    sub = 20;
end
filename=['study2_4vs', mat2str(layer), '_RSA_comparison', char(type), '.emf'];

%% 画图
maindir=pwd;
alllayer=6;

dir = strcat('F:\TE_DCNN_RSA\RDMs with EEG\study2_resnet-50\IN\', type, '\all\fieldtriped\');
IN_N_C = load(strcat(dir, "Neuro_resnet50_", type, "_all")).N_C;
dir = strcat('F:\TE_DCNN_RSA\RDMs with EEG\study2_resnet-50\SIN\', type, '\all\fieldtriped\');
SIN_N_C = load(strcat(dir, "Neuro_resnet50_", type, "_all")).N_C;
dir = strcat('F:\TE_DCNN_RSA\RDMs with EEG\study2_resnet-50\SININ\', type, '\all\fieldtriped\');
SININ_N_C = load(strcat(dir, "Neuro_resnet50_", type, "_all")).N_C;
for row = 1:sub
    N_C_IN{row, 1} = IN_N_C{row, 1};
    N_C_SIN{row, 1} = SIN_N_C{row, 1};
    N_C_SININ{row, 1} = SININ_N_C{row, 1};
end

for row = 1:size(N_C_IN, 1)
    IN{1, row}.avg = N_C_IN{row, 1}(layer, [1:(sampling*1000):471]);
    IN{1, row}.avg(2, :) = N_C_IN{row, 1}(layer, [1:(sampling*1000):471]);
    IN{1, row}.time = (-0.1:sampling:0.371);
    IN{1, row}.dimord = 'chan_time';
    IN{1, row}.label(1, 1) = {['P1']};
    IN{1, row}.label(2, 1) = {['P2']};

    SIN{1, row}.avg = N_C_SIN{row, 1}(layer, [1:(sampling*1000):471]);
    SIN{1, row}.avg(2, :) = N_C_SIN{row, 1}(layer, [1:(sampling*1000):471]);
    SIN{1, row}.time = (-0.1:sampling:0.371);
    SIN{1, row}.dimord = 'chan_time';
    SIN{1, row}.label(1, 1) = {['P1']};
    SIN{1, row}.label(2, 1) = {['P2']};

    SININ{1, row}.avg = N_C_SININ{row, 1}(layer, [1:(sampling*1000):471]);
    SININ{1, row}.avg(2, :) = N_C_SININ{row, 1}(layer, [1:(sampling*1000):471]);
    SININ{1, row}.time = (-0.1:sampling:0.371);
    SININ{1, row}.dimord = 'chan_time';
    SININ{1, row}.label(1, 1) = {['P1']};
    SININ{1, row}.label(2, 1) = {['P2']};

end

% model间
significant_timepause_model_contrast = func_between_models_Ttest(IN, SIN, SININ, type, sampling, layer, sided_for_model, alpha, correct_for_model, seed);
significant_timepause_model_contrast = significant_timepause_model_contrast*sampling*1000;

% layer间
if layer == 5
    significant_timepause_4_vs_5 = func_layer_4_5_Ttest(N_C_IN, sided_for_layers, alpha, correct_for_layers, seed, sampling);
    ST_IN = significant_timepause_4_vs_5; clear significant_timepause_4_vs_5;
    
    
    significant_timepause_4_vs_5 = func_layer_4_5_Ttest(N_C_SIN, sided_for_layers, alpha, correct_for_layers, seed, sampling);
    ST_SIN = significant_timepause_4_vs_5; clear significant_timepause_4_vs_5;
    
    significant_timepause_4_vs_5 = func_layer_4_5_Ttest(N_C_SININ, sided_for_layers, alpha, correct_for_layers, seed, sampling);
    ST_SININ = significant_timepause_4_vs_5; clear significant_timepause_4_vs_5;
    
    significant_timepause_4_vs_5 = {};
    if size(ST_IN,1)>=1
        significant_timepause_4_vs_5{1,1} = ST_IN{1,1};
    end
    if size(ST_SIN,1)>=1
        significant_timepause_4_vs_5{2,1} = ST_SIN{1,1};
    end
    if size(ST_SININ,1)>=1
        significant_timepause_4_vs_5{3,1} = ST_SININ{1,1};
    end
    significant_timepause_between_layers = significant_timepause_4_vs_5 * sampling * 1000;
elseif layer == 6
    significant_timepause_4_vs_6 = func_layer_4_6_Ttest(N_C_IN, sided_for_layers, alpha, correct_for_layers, seed, sampling);
    ST_IN = significant_timepause_4_vs_6; clear significant_timepause_4_vs_6;
    
    significant_timepause_4_vs_6 = func_layer_4_6_Ttest(N_C_SIN, sided_for_layers, alpha, correct_for_layers, seed, sampling);
    ST_SIN = significant_timepause_4_vs_6; clear significant_timepause_4_vs_6;
    
    significant_timepause_4_vs_6 = func_layer_4_6_Ttest(N_C_SININ, sided_for_layers, alpha, correct_for_layers, seed, sampling);
    ST_SININ = significant_timepause_4_vs_6; clear significant_timepause_4_vs_6;
    
    significant_timepause_4_vs_6 = {};
    if size(ST_IN,1)>=1
        significant_timepause_4_vs_6{1,1} = ST_IN{1,1};
    end
    if size(ST_SIN,1)>=1
        significant_timepause_4_vs_6{2,1} = ST_SIN{1,1};
    end
    if size(ST_SININ,1)>=1
        significant_timepause_4_vs_6{3,1} = ST_SININ{1,1};
    end
    significant_timepause_between_layers = significant_timepause_4_vs_6 * sampling * 1000;
end

for j = 1:size(N_C_IN, 1)
    N_C{j,1}(1, :) = N_C_IN{j,1}(4, :);
    N_C{j,1}(2, :) = N_C_IN{j,1}(layer, :);
    N_C{j,1}(3, :) = N_C_SIN{j,1}(4, :);
    N_C{j,1}(4, :) = N_C_SIN{j,1}(layer, :);
    N_C{j,1}(5, :) = N_C_SININ{j,1}(4, :);
    N_C{j,1}(6, :) = N_C_SININ{j,1}(layer, :);
end

% layer vs 0
significant_timepause = func_Ttest(N_C_IN, sided_for_layer_vs_0, alpha, correct_for_layer_vs_0, seed, sampling);
ST_IN = significant_timepause; clear significant_timepause;

significant_timepause = func_Ttest(N_C_SIN, sided_for_layer_vs_0, alpha, correct_for_layer_vs_0, seed, sampling);
ST_SIN = significant_timepause; clear significant_timepause;

significant_timepause = func_Ttest(N_C_SININ, sided_for_layer_vs_0, alpha, correct_for_layer_vs_0, seed, sampling);
ST_SININ = significant_timepause; clear significant_timepause;

if size(ST_IN,1)>=layer
    significant_timepause{1,1} = ST_IN{4,1};
    significant_timepause{2,1} = ST_IN{layer,1};
elseif size(ST_IN,1)>=4
    significant_timepause{1,1} = ST_IN{4,1};
end
if size(ST_SIN,1)>=layer
    significant_timepause{3,1} = ST_SIN{4,1};
    significant_timepause{4,1} = ST_SIN{layer,1};
elseif size(ST_SIN,1)>=4
    significant_timepause{3,1} = ST_SIN{4,1};
end
if size(ST_SININ,1)>=layer
    significant_timepause{5,1} = ST_SININ{4,1};
    significant_timepause{6,1} = ST_SININ{layer,1};
elseif size(ST_SININ,1)>=4
    significant_timepause{5,1} = ST_SININ{4,1};
end
significant_timepause = significant_timepause * sampling * 1000;

for v=1:alllayer
    rn50_all{v}=N_C{1,1}(v, :);
    for s=1:(length(N_C)-1)
        rn50_all{v}=vertcat(rn50_all{v},N_C{s+1,1}(v, :));
    end
end


%% 画图
x=1:471;
y=zeros(1,471);
xp=[x fliplr(x)]; 

colormap=[129 207 244; 244 175 179; 1 145 198; 220 101 162; 30 56 147; 209 45 29];
colormap_layercontrast=[169 167 236; 169 55 255; 99 55 255];
colormap_modelcontrast=[227 218 47;224 162 70;224 129 70];% IN SIN, IN SININ, SIN SININ

figure,
for v=1:alllayer %8layer
    eval(['Y' mat2str(v) '=mean(rn50_all{v});']);%平均每个组的相关
    eval(['E' mat2str(v) '= std(rn50_all{v})/sqrt(length(N_C));']);%求每个组的标准误%shape
end 

% layer
for v=1:alllayer
    eval(['yp' mat2str(v) '=[Y' mat2str(v) '-E' mat2str(v) ' fliplr(Y' mat2str(v) '+E' mat2str(v) ')];']); 
    %Y-E是阴影下?? fliplr（Y+E）阴影上??
    eval(['yp=yp' mat2str(v)]);
    fill(xp,yp,'k','Linestyle','none','FaceColor',[colormap(v,:)/255],'Facealpha',0.1);%0,158,225
    hold on
end

init1 = 0.075;
init2 = -0.04;
init3 = 0.1;
%画平均线
for v=1:alllayer
    eval(['Y=Y' mat2str(v)]);
    plot(Y,'Color',[colormap(v,:)/255],'Linewidth',1);
    hold on
    % layer间
    if(v <= size(significant_timepause_between_layers,1))
        if(isempty(significant_timepause_between_layers{v, 1}) == 0)
            nums = significant_timepause_between_layers{v, 1};
            % 将单元格中的值成对取出
            for j = 1:2:numel(nums)
                num1 = nums(j);
                num2 = nums(j+1);
                % 如果是正相关，画线
                mid = round((num1+num2)/2);
                line([num1,num2],[init1,init1],'Color',colormap_layercontrast(v,:)/255,'linewidth', 2,'linestyle','-');
            end
        end
    end
    init1 = init1 + 0.003;
    % layer vs 0
    if(v <= size(significant_timepause,1))
        if(isempty(significant_timepause{v, 1}) == 0)
            nums = significant_timepause{v, 1};
            % 将单元格中的值成对取出
            for j = 1:2:numel(nums)
                num1 = nums(j);
                num2 = nums(j+1);
                mid = round((num1+num2)/2);
                line([num1,num2],[init2,init2],'Color',colormap(v,:)/255,'linewidth', 2,'linestyle','-');
            end
        end
    end
    init2 = init2 + 0.003;
    % model间
    if(v <= size(significant_timepause_model_contrast,1))
        if(isempty(significant_timepause_model_contrast{v, 1}) == 0)
            nums = significant_timepause_model_contrast{v, 1};
            % 将单元格中的值成对取出
            for j = 1:2:numel(nums)
                num1 = nums(j);
                num2 = nums(j+1);
                mid = round((num1+num2)/2);
                line([num1,num2],[init3,init3],'Color',colormap_modelcontrast(v,:)/255,'linewidth', 2,'linestyle','-');
            end
        end
    end
    init3 = init3 + 0.003;
end

%0??-虚线
plot(x,y,'LineStyle','--','Color',[0 0 0]);
xlabel({'Onset time of the sliding window (ms)'});
ylabel({'Correlation (Zscore)'});
axis([0,500,-0.055,0.15]);

cd(maindir);
saveas(gcf,filename);


