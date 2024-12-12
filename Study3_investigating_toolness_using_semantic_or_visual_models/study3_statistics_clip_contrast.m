clear;

clip_imageencoder = "resnet50";  % resnet50 or vit
type = "visible";  % cfs or visible
sided = 1;
alpha = 0.05;
sampling = 0.001;
if type == "cfs"
    sub = 21;
elseif type == "visible"
    sub = 20;
end

%% 整理数据
dir = strcat('F:\TE_DCNN_RSA\RDMs with EEG\study3_sementic_vs_image\BERT\base\', type, '\');
BERT_N_C = load(strcat(dir, "Neuro_", type, "_all")).N_C;
dir = strcat('F:\TE_DCNN_RSA\RDMs with EEG\study3_sementic_vs_image\ViT\ViT16\base\', type, '\');
ViT_N_C = load(strcat(dir, "Neuro_", type, "_all")).N_C;

if clip_imageencoder == "resnet50"
    dir = strcat(['F:\TE_DCNN_RSA\RDMs with EEG\study3_sementic_vs_image\CLIP\RN50\' ...
        'extract_from_output\'], type, '\two-sided\fieldtriped\all\');
    CLIP_N_C = load(strcat(dir, "Neuro_resnet50_", type, "_all")).N_C;
elseif clip_imageencoder == "vit"
    dir = strcat('F:\TE_DCNN_RSA\RDMs with EEG\study3_sementic_vs_image\CLIP\vit16\', type, '\all\');
    CLIP_N_C = load(strcat(dir, "Neuro_", type, "_all")).N_C;
end

for row = 1:size(CLIP_N_C, 1)
    CLIP_TextEncoder{1, row}.avg = CLIP_N_C{row, 1}(2, :);
    CLIP_TextEncoder{1, row}.avg(2, :) = CLIP_N_C{row, 1}(2, :);
    CLIP_TextEncoder{1, row}.time = (-0.1:sampling:0.371);
    CLIP_TextEncoder{1, row}.dimord = 'chan_time';
    CLIP_TextEncoder{1, row}.label(1, 1) = {['P1']};
    CLIP_TextEncoder{1, row}.label(2, 1) = {['P2']};

    ClIP_ImageEncoder{1, row}.avg = CLIP_N_C{row, 1}(1, :);
    ClIP_ImageEncoder{1, row}.avg(2, :) = CLIP_N_C{row, 1}(1, :);
    ClIP_ImageEncoder{1, row}.time = (-0.1:sampling:0.371);
    ClIP_ImageEncoder{1, row}.dimord = 'chan_time';
    ClIP_ImageEncoder{1, row}.label(1, 1) = {['P1']};
    ClIP_ImageEncoder{1, row}.label(2, 1) = {['P2']};

    BERT{1, row}.avg = BERT_N_C{row, 1}(1, :);
    BERT{1, row}.avg(2, :) = BERT_N_C{row, 1}(1, :);
    BERT{1, row}.time = (-0.1:sampling:0.371);
    BERT{1, row}.dimord = 'chan_time';
    BERT{1, row}.label(1, 1) = {['P1']};
    BERT{1, row}.label(2, 1) = {['P2']};


    ViT{1, row}.avg = ViT_N_C{row, 1}(1, :);
    ViT{1, row}.avg(2, :) = ViT_N_C{row, 1}(1, :);
    ViT{1, row}.time = (-0.1:sampling:0.371);
    ViT{1, row}.dimord = 'chan_time';
    ViT{1, row}.label(1, 1) = {['P1']};
    ViT{1, row}.label(2, 1) = {['P2']};

end


%% 两两比较
ft_defaults
cfg = [];
% cfg.channel     = 'all';
cfg.latency     = [-0.1 0.371];
cfg.avgoverchan = 'yes';
cfg.parameter   = 'avg';
cfg.method      = 'montecarlo';%all
cfg.statistic   = 'ft_statfun_depsamplesT';
cfg.alpha       = alpha;
cfg.tail        = sided;
cfg.correctm    = 'cluster';
cfg.correcttail = 'prob';
cfg.numrandomization = 1000;
subj = sub;
design = zeros(2,2*subj);
for i = 1:subj
design(1,i) = i;
end
for i = 1:subj
design(1,subj+i) = i;
end
design(2,1:subj)        = 1;
design(2,subj+1:2*subj) = 2;

cfg.design = design;
cfg.uvar  = 1;
cfg.ivar  = 2;


stat_semantic = ft_timelockstatistics(cfg, CLIP_TextEncoder{:}, BERT{:});
stat_visual = ft_timelockstatistics(cfg, ClIP_ImageEncoder{:}, ViT{:});

%% 识别显著区间
if min(stat_semantic.prob)<=0.05
    significant_time_semantic=find(stat_semantic.mask==1);  
    significant_timepause_semantic(1)=significant_time_semantic(1);
    k=2;
    for j=2:length(significant_time_semantic)-1
        if significant_time_semantic(j-1)==significant_time_semantic(j)-1 &&...
                significant_time_semantic(j+1)==significant_time_semantic(j)+1
            significant_timepause_semantic(k)=0;
        else
            significant_timepause_semantic(k)=significant_time_semantic(j);
            k=k+1;
        end
    end
     significant_timepause_semantic(k)=significant_time_semantic(length(significant_time_semantic));
end 

if min(stat_visual.prob)<=0.05
    significant_time_visual=find(stat_visual.mask==1);  
    significant_timepause_visual(1)=significant_time_visual(1);
    k=2;
    for j=2:length(significant_time_visual)-1
        if significant_time_visual(j-1)==significant_time_visual(j)-1 &&...
                significant_time_visual(j+1)==significant_time_visual(j)+1
            significant_timepause_visual(k)=0;
        else
            significant_timepause_visual(k)=significant_time_visual(j);
            k=k+1;
        end
    end
     significant_timepause_visual(k)=significant_time_visual(length(significant_time_visual));
end 

