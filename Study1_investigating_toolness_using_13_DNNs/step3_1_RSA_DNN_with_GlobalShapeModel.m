clear;

% DSM from AlexNet
load('RDMvector');

%% generate global shape model matrix
% 1-20：ET; 21-40:ST; 41-60:ENT; 61-80:SNT
shapemodelRDM = zeros(80, 80);
shapemodelRDM(1:20, 21:40) = 1;
shapemodelRDM(21:40, 1:20) = 1;
shapemodelRDM(1:20, 61:80) = 1;
shapemodelRDM(61:80, 1:20) = 1;
shapemodelRDM(21:40, 41:60) = 1;
shapemodelRDM(41:60, 21:40) = 1;
shapemodelRDM(41:60, 61:80) = 1;
shapemodelRDM(61:80, 41:60) = 1;

i = 0;
for firststim = 1:79
    for secstim = (firststim + 1):80
        i = i + 1;
        stimpair_Order(i, 1) = i;  
        stimpair_Order(i, 2) = firststim;  
        stimpair_Order(i, 3) = secstim;  
    end
end
for i = 1:size(stimpair_Order, 1)
    shapemodel_vector(i, 1) = shapemodelRDM(stimpair_Order(i, 2), stimpair_Order(i, 3));
end
save('shapemodelRDM', 'shapemodelRDM');
save('shapemodel_vector', 'shapemodel_vector');

%% get DSM of global shape
load('shapemodel_vector');
cd('Shape');
RDMshape_alexnet = [];
for i = 1:size(RDMvector, 2)
    A = shapemodel_vector(:, 1);
    B = double(RDMvector(:, i));
    RDMshape_alexnet(i, 1)= corr(A, B,'Type','Spearman','Rows','complete');
end
save('RDMshape_alexnet','RDMshape_alexnet');

%% figure
% trained_vs_nontrained-nontrained
color=[0 0 0;105 105 105;135 206 250;255 165 0;255 99 71;255 0 0]/255;
figure;
b = bar(RDMshape_alexnet, 'BarWidth', 0.8)
b.FaceColor = 'flat';
for v=1:6
    b.CData(v,:) = color(v,:);
end
set(gca, 'YLim', [0 0.4])
set(gca, 'YTick', 0:0.05:0.4)
ylabel('Correlation');
xlabel('Layer');
saveas(gcf, 'RDMshape_alexnet.png');
