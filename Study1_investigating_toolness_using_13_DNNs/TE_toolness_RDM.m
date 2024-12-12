clear;

load('TEvisible_toolquestionnaire_allsubjs');
load('TECFS_toolquestionnaire_allsubjs');
load('TEBM_toolquestionnaire_allsubjs');

load('RDMvector');

toolnessRDM(:, 1:20) = TEvisible_toolquestionnaire(:, :);
toolnessRDM(:, 21:41) = TECFS_toolquestionnaire(:, :);
toolnessRDM(:, 42:61) = TEBM_toolquestionnaire(:, :);
%% ËãRDM
for sub = 1:size(toolnessRDM, 2)
    i=0;
    for firststim=1:79
        for secstim=(firststim+1):80
            i=i+1;
            stimpair_Order(i,1)=i;  %ï¿½ï¿½Ô´Ì¼ï¿½ï¿½ï¿½Ë³ï¿½ï¿?
            stimpair_Order(i,2)=firststim;  %ï¿½ï¿½Ò»ï¿½ï¿½Í¼ï¿½ï¿½ï¿?
            stimpair_Order(i,3)=secstim;  %ï¿½Ú¶ï¿½ï¿½ï¿½Í¼ï¿½Ä±ï¿½ï¿?
        end
    end
    % ï¿½ï¿½ï¿½RDMï¿½ï¿½RDMvector
    for i = 1:size(stimpair_Order, 1)
        A = toolnessRDM(stimpair_Order(i, 2), sub);
        B = toolnessRDM(stimpair_Order(i, 3), sub);
        RDMtoolness{sub}(stimpair_Order(i, 2), stimpair_Order(i, 3)) = abs(A - B);
        RDMtoolness{sub}(stimpair_Order(i, 3), stimpair_Order(i, 2)) = RDMtoolness{sub}(stimpair_Order(i, 2), stimpair_Order(i, 3));
        RDMvector_toolness(i, sub) = RDMtoolness{sub}(stimpair_Order(i, 2), stimpair_Order(i, 3));
    end
end
save('RDMtoolness', 'RDMtoolness');
save('RDMvector_toolness', 'RDMvector_toolness');

%% ËãRDM
RDMtoolness_results_rn50 = [];
cd('Toolness');
for i = 1:size(toolnessRDM, 2)
    for j = 1:size(RDMvector, 2)
        A = RDMvector_toolness(:, i);
        B = double(RDMvector(:, j));
        RDMtoolness_results_rn50(j, i)= corr(A, B,'Type','Spearman','Rows','complete');
    end
end
save('RDMtoolness_results_rn50','RDMtoolness_results_rn50');

%% »­Í¼
color=[0 0 0;105 105 105;135 206 250;255 165 0;255 99 71;255 0 0]/255;
avg = mean(RDMtoolness_results_rn50, 2);
x = 1:6;
b = bar(x, avg);
set(b, 'BarWidth', 0.7);
hold on
b.FaceColor = 'flat';
for v = 1:6
    b.CData(v,:) = color(v,:);
end
sub = length(RDMtoolness_results_rn50);
for v = 1:6
	SEM(v, 1) = std(RDMtoolness_results_rn50(v, :))/sqrt(sub);
end       
ngroups = size(RDMtoolness_results_rn50, 2);
nbars = size(RDMtoolness_results_rn50);
for k = 1:6
    err = errorbar(x(k), avg(k, 1), SEM(k, 1), 'x', 'linewidth',2);
    set(err,'Color',color(k, :));
end
err.LineStyle = 'none'; 
hold on;
set(gca,'XTickLabel',{'1','2','3','4','5','6'});
set(gca, 'YLim', [-0.08 0.14])
set(gca, 'YTick', -0.08:0.02:0.14)
ylabel('Correlation');
saveas(gcf, 'RDMtoolness_results_rn50.png');
