%% DSM
clear;
load('Resnet50LayersOutput');

i=0;
for firststim=1:79
    for secstim=(firststim+1):80
        i=i+1;
        stimpair_Order(i,1)=i;  %��Դ̼���˳��
        stimpair_Order(i,2)=firststim;  %��һ��ͼ���
        stimpair_Order(i,3)=secstim;  %�ڶ���ͼ�ı��
    end
end

for L = 1:size(layer, 2)  
    % ÿһ����80��ͼ֮������,�γ�80x80�Ĳ������Ծ���
    for i = 1:size(stimpair_Order, 1)
        A = layer{stimpair_Order(i, 2), L};
        stim1 = reshape(A,size(A,1)*size(A,3)*size(A,2),1);
        B = layer{stimpair_Order(i, 3), L};
        stim2 = reshape(B,size(B,1)*size(B,3)*size(B,2),1);
        RDM{L}(stimpair_Order(i, 2), stimpair_Order(i, 3)) = 1 - corr(stim1, stim2, 'type', 'Spearman', 'rows' , 'complete');
        RDM{L}(stimpair_Order(i, 3), stimpair_Order(i, 2)) = RDM{L}(stimpair_Order(i, 2), stimpair_Order(i, 3));
        RDMvector(i,L) =  RDM{L}(stimpair_Order(i, 2), stimpair_Order(i, 3));
    end
    waitbar(i/80, h);
    %save(strcat('RDM_',num2str(i)), 'RDM');
end
save('RDM', 'RDM');
save('RDMvector', 'RDMvector');

%% ��ͼ
for i = 1:size(layer, 2)
    load('RDM');
    imagesc(RDM{i});
    colormap(othercolor(('BuDRd_18')))
    colorbar%("ylim", [0,1])
    title(strcat('Layer_', num2str(i)));
    set(gca, 'CLim', [0, 1])
    saveas(gcf, strcat('resnet50layer_',num2str(i),'.emf'));
end