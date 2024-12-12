clear;

type = "cfs";  % cfs or visible
sided = 1;
alpha = 0.05;
sampling = 0.001;
if type == "cfs"
    sub = 21;
elseif type == "visible"
    sub = 20;
end

%% 整理数据
dir = strcat('F:\TE_DCNN_RSA\RDMs with EEG\study3_sementic_vs_image\BERT\base\output\', type, '\all\');
BERT_N_C = load(strcat(dir, "Neuro_", type, "_all")).N_C;
dir = strcat('F:\TE_DCNN_RSA\RDMs with EEG\study3_sementic_vs_image\ViT\ViT16\base\', type, '\');
ViT_N_C = load(strcat(dir, "Neuro_", type, "_all")).N_C;

for row = 1:size(BERT_N_C, 1)
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
% cfg.latency     = [-0.1 0.371];
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


stat = ft_timelockstatistics(cfg, ViT{:}, BERT{:});

%% 识别显著区间
if min(stat.prob)<=0.05
    significant_time=find(stat.mask==1);  
    significant_timepause{1,1}(1,1)=significant_time(1);
    k=2;
    for j=2:length(significant_time)-1
        if significant_time(j-1)==significant_time(j)-1 &&...
                significant_time(j+1)==significant_time(j)+1
            significant_timepause{1,1}(1,k)=0;
        else
            significant_timepause{1,1}(1,k)=significant_time(j);
            k=k+1;
        end
    end
     significant_timepause{1,1}(1,k)=significant_time(length(significant_time));
end 


