clear;close;
%%加载数据

load('m1d00');load('m2d00');load('m3d00');load('m4d00');

train_mode1_norm=m1d00(1:500,1:54)';
train_mode2_norm=m2d00(1:500,1:54)';
train_mode3_norm=m3d00(1:500,1:54)';
train_mode4_norm=m4d00(1:500,1:54)';

train_mode_set={train_mode1_norm;train_mode2_norm;train_mode3_norm;train_mode4_norm};

%将四个矩阵按照行拼接在一起
train_mode_norm=[train_mode1_norm;train_mode2_norm;train_mode3_norm;train_mode4_norm];
%标准化数据，使用zscore函数
data_normalized=zscore(train_mode_norm);%data_normalized为标准化后的数据矩阵
% 使用AHC聚类

% Step 1: 计算距离矩阵（欧几里得距离）
euclidean_distancae = pdist(data_normalized, 'euclidean');  % 'euclidean' 计算欧几里得距离

% Step 2: 使用 linkage 函数进行层次聚类
% 使用 'single' 表示单链法（Single Linkage），也可以使用 'complete', 'average' 等其他方法
C_tree = linkage(euclidean_distancae, 'single');  % 返回的 Z 是一个层次聚类树

% Step 3: 绘制树状图（dendrogram），以帮助选择聚类的数量
dendrogram(C_tree);  % 可以看到树状图

% Step 4: 根据树状图确定聚类的数量，假设我们选择了 S 个簇
mdoe_nember = 4;  % 假设我们决定选择 3 个簇

% Step 5: 切割树状图，获得最终的聚类标签
T = cluster(C_tree, 'maxclust', mdoe_nember);  % 'maxclust' 用于指定簇的数量

%%计算模态内邻接矩阵 拉普拉斯矩阵
W_intra=cell(mdoe_nember,1);
L_intra=cell(mdoe_nember,1);
W_D=cell(mdoe_nember,1);
W_A=cell(mdoe_nember,1);
train_mode1_norm_for_L=m1d00(1:500,1:54);
train_mode2_norm_for_L=m2d00(1:500,1:54);
train_mode3_norm_for_L=m3d00(1:500,1:54);
train_mode4_norm_for_L=m4d00(1:500,1:54);
train_mode_set_for_L={train_mode1_norm_for_L;train_mode2_norm_for_L;train_mode3_norm_for_L;train_mode4_norm_for_L};
for i=1:mdoe_nember
    % 计算模态内邻接矩阵
    [W_D{i},W_A{i}]=compute_intra_adjacency_matrix(train_mode_set_for_L{i},10,1,1);
    W_intra{i}=W_D{i}+W_A{i};
    % 计算拉普拉斯矩阵
    L_intra{i}=compute_laplacian(W_intra{i});
end

%%开始优化变量，高兴Z，P，J，E
lower_dim=10; %设定降维后的维度
%获得训练数据的维度
[dim,N_samples_of_all]=size(train_mode_norm);%N_samples_of_all为总样本数，dim为特征数

X=train_mode_norm; %训练数据矩阵X
%初始化变量Z,N_samples_of_all*N_samples_of_all,Z为公共的低秩矩阵，不用堆叠
Z=zeros(N_samples_of_all,N_samples_of_all);
%初始化变量E,E的大小为dim*(N_samples_of_all)
E=zeros(dim,N_samples_of_all);
%初始化变量J,J的大小为N_samples_of_all*N_samples_of_all,J为公共的低秩矩阵Z的辅助变量，不用堆叠
J=zeros(N_samples_of_all,N_samples_of_all);
%初始化变量P,P的大小为(N_samples_of_all)*lower_dim
P=zeros(dim,lower_dim);
%初始化辅助变量 \Gamma，大小和train_mode_norm一致
Gamma=zeros(dim,N_samples_of_all);
%初始化变量 \Lambda，大小和Z一致
Lambda=zeros(N_samples_of_all,N_samples_of_all);

%%开始更新变量,迭代更新
max_iter=50; %最大迭代次数
tolerance=1e-6; %收敛阈值
%初始化收敛标志位
converged = false;
%迭代更新变量
mu = 1e-6; % 参数 mu
mu_max = 1e6; % 最大 mu
rho = 1.1; % 参数 rho
I = eye(N_samples_of_all);  % 生成单位矩阵，维度是 dim
lambda = 1; % 例如，lambda 可能是给定的正则化常数
gamma = 1;  % 设置 gamma
beta = 1;   % 设置 beta
alpha = 1;    % 假设 alpha 的值
L_total = zeros(size(L_intra{1}));
for i = 1:mdoe_nember
    L_total = L_total + L_intra{i};  % 累加每个模态的拉普拉斯矩阵
end


% 初始化记录变量收敛的数组
Z_norm = zeros(max_iter, 1);
E_norm = zeros(max_iter, 1);
P_norm = zeros(max_iter, 1);
J_norm = zeros(max_iter, 1);
Gamma_norm = zeros(max_iter, 1);
Lambda_norm = zeros(max_iter, 1);
mu_values = zeros(max_iter, 1);


for iter=1:max_iter
    %%更新J

    threshold = 1 / mu;  % 软阈值参数

    J = soft_thresholding(Z + Lambda / rho, threshold);
    %%更新P
    Z_minus_I = Z - I;
    Z_diff = Z_minus_I * Z_minus_I';
    A = X*(gamma * Z_diff + beta * L_total)*X';
    [eigVectors, eigValues] = eig(A);  % 求解矩阵 A 的特征值和特征向量
    [~, idx] = sort(diag(eigValues), 'descend');  % 根据特征值排序
    P = eigVectors(:, idx(1:lower_dim));  % 选择最大的 lower_dim 个特征向量


    %%更新Z
    % 计算 Z_{k+1}
    term1 = lambda * X' * (P * P') * X + mu * (X' * X) + mu * I; % 第一部分
    term2 = lambda * X' * (P * P') * X + mu * (X' * X) - mu * X' * E + mu * J + X' * Gamma - Lambda; % 第二部分

    % 使用反斜杠运算符进行矩阵求解
    Z = term1 \ term2;
    %%更新E
    Q = X - X * Z + Gamma / mu;
    for i = 1:N_samples_of_all
        norm_Q = norm(Q(:, i), 2);  % 计算 Q 的第 i 列的 L2 范数
        if norm_Q > alpha
            % 如果 L2 范数大于 alpha，按公式更新第 i 列
            E(:, i) = (norm_Q - alpha) / norm_Q * Q(:, i);
        else
            % 否则，将 E 的第 i 列设置为 0
            E(:, i) = 0;
        end
    end
    %%更新辅助变量和参数
    Gamma = Gamma +  mu* (X - X * Z - E);
    Lambda = Lambda + mu * (Z - J);
    mu = min(rho * mu, mu_max);

    %% 记录每个变量的范数以便可视化收敛过程
    Z_norm(iter) = norm(Z, 'fro');
    E_norm(iter) = norm(E, 'fro');
    P_norm(iter) = norm(P, 'fro');
    J_norm(iter) = norm(J, 'fro');
    Gamma_norm(iter) = norm(Gamma, 'fro');
    Lambda_norm(iter) = norm(Lambda, 'fro');
    mu_values(iter) = mu;
    
end
% 可视化每个变量的收敛过程
figure;

subplot(3, 2, 1);
plot(1:max_iter, Z_norm);
title('Convergence of Z');
xlabel('Iteration');
ylabel('Norm of Z');

subplot(3, 2, 2);
plot(1:max_iter, E_norm);
title('Convergence of E');
xlabel('Iteration');
ylabel('Norm of E');

subplot(3, 2, 3);
plot(1:max_iter, P_norm);
title('Convergence of P');
xlabel('Iteration');
ylabel('Norm of P');

subplot(3, 2, 4);
plot(1:max_iter, J_norm);
title('Convergence of J');
xlabel('Iteration');
ylabel('Norm of J');

subplot(3, 2, 5);
plot(1:max_iter, Gamma_norm);
title('Convergence of Gamma');
xlabel('Iteration');
ylabel('Norm of Gamma');

subplot(3, 2, 6);
plot(1:max_iter, Lambda_norm);
title('Convergence of Lambda');
xlabel('Iteration');
ylabel('Norm of Lambda');

% 可视化 mu 的收敛过程
figure;
plot(1:max_iter, mu_values);
title('Convergence of mu');
xlabel('Iteration');
ylabel('mu');

function J_new = soft_thresholding(Z_lambda, threshold)
        % 对矩阵 Z_lambda 的奇异值进行软阈值操作
        [U, S, V] = svd(Z_lambda, 'econ');  % 计算奇异值分解
        S = max(S - threshold, 0);  % 软阈值处理
        J_new = U * S * V';  % 重构矩阵
end

