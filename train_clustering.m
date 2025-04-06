clear;close;
%%加载数据

load('m1d00');load('m2d00');load('m3d00');load('m4d00');

train_mode1_norm=m1d00(1:200,1:54);
train_mode2_norm=m2d00(1:200,1:54);
train_mode3_norm=m3d00(1:200,1:54);
train_mode4_norm=m4d00(1:200,1:54);

train_mode_set={train_mode1_norm;train_mode2_norm;train_mode3_norm;train_mode4_norm};

%将四个矩阵按照行拼接在一起
train_mode_norm=[train_mode1_norm;train_mode2_norm;train_mode3_norm;train_mode4_norm];

%标准化数据，使用zscore函数
data_normalized=zscore(train_mode_norm);%data_normalized为标准化后的数据矩阵
% 使用AHC聚类

% Step 1: 计算距离矩阵（欧几里得距离）
euclidean_distancae = pdist(train_mode_norm, 'euclidean');  % 'euclidean' 计算欧几里得距离

% Step 2: 使用 linkage 函数进行层次聚类
% 使用 'single' 表示单链法（Single Linkage），也可以使用 'complete', 'average' 等其他方法
C_tree = linkage(euclidean_distancae, 'single');  % 返回的 Z 是一个层次聚类树

% Step 3: 绘制树状图（dendrogram），以帮助选择聚类的数量
% dendrogram(C_tree);  % 可以看到树状图

% Step 4: 根据树状图确定聚类的数量，假设我们选择了 S 个簇
mdoe_nember = 4;  % 假设我们决定选择 3 个簇

% Step 5: 切割树状图，获得最终的聚类标签
T = cluster(C_tree, 'maxclust', mdoe_nember);  % 'maxclust' 用于指定簇的数量

% 绘制在特征1和特征2上的聚类结果
figure;
gscatter(data_normalized(:, 1), data_normalized(:, 2), T, 'rbgy', 'o', 8);  % 使用列作为 X 和 Y
title('Clustering Results on Feature 1 and Feature 2');
xlabel('Feature 1');
ylabel('Feature 2');
legend('Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4');
grid on;
% 绘制在特征3和特征5上的聚类结果
figure;
gscatter(data_normalized(:, 3), data_normalized(:, 5), T, 'rbgy', 'o', 8);  % 使用列作为 X 和 Y
title('Clustering Results on Feature 3 and Feature 5');
xlabel('Feature 3');
ylabel('Feature 5');
legend('Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4');
grid on;


