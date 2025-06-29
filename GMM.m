function [Label, mu] = GMM(Problem, Fitness, K)
    %% Initialization and parameter setting
    M = Problem.M;
    N1 = size(Fitness,1);
    N_pdf=zeros(N1, K); % 存储每个样本对每个高斯分量的概率密度值 (N1×K矩阵)
    para_sigma_inv=zeros(M, M, K); % 存储每个高斯分量协方差矩阵的逆 (M×M×K张量)
    RegularizationValue = 0.001; % 正则化参数（防止协方差矩阵奇异）
    MaxIter = 300; % EM算法最大迭代次数
    TolFun = 1e-8; % 收敛容差（似然函数变化阈值）

    %% 构建高斯混合模型 (GMM)
    % K: 高斯分量数量
    % CovarianceType: 'diagonal'：使用对角协方差矩阵（假设特征独立）, Start: 'plus'：用k-means++算法初始化参数
    % RegularizationValue：协方差矩阵对角线正则化值, MaxIter=300：最大EM迭代次数
    % TolFun=1e-8：似然函数变化收敛阈值
    gmm = fitgmdist(Fitness,K,'RegularizationValue',RegularizationValue,'CovarianceType','diagonal','Start','plus','Options',statset('Display','final','MaxIter',MaxIter,'TolFun',TolFun));
    
    % 提取模型参数
    mu = gmm.mu; % 各高斯分量的均值矩阵 (K×M)
    sigma = gmm.Sigma; % 协方差矩阵 (M×M×K张量)
    ComponentProportion = gmm.ComponentProportion; % 各分量的混合权重 (1×K向量)

    %% 计算概率，进行聚类
    % 计算协方差矩阵的逆
    for k=1:K
        % 由于使用对角协方差，求逆简化为对对角线元素取倒数
        sigma_inv=1./sigma(:,:,k);  % 对对角矩阵求逆（元素级倒数）, sigma的逆矩阵,(X_dim, X_dim)的矩阵
        para_sigma_inv(:, :, k)=diag(sigma_inv);  % 存储为对角矩阵形式 sigma^(-1)
    end

    % 计算概率密度 (N_pdf)
    for k=1:K
        % 计算概率密度函数的归一化系数
        coefficient=(2*pi)^(-M/2)*sqrt(det(para_sigma_inv(:, :, k)));  % 高斯分布的概率密度函数e左边的系数
        
        % 计算样本与均值的偏差
        X_miu=Fitness-repmat(mu(k,:), N1, 1);  % X-miu: (X_num, X_dim)的矩阵

        % 计算指数部分
        exp_up=sum((X_miu*para_sigma_inv(:, :, k)).*X_miu,2);  % 指数的幂，(X-miu)'*sigma^(-1)*(X-miu)
        
        % 计算完整概率密度
        N_pdf(:,k)=coefficient*exp(-0.5*exp_up);
    end

    %% 计算后验概率，以及归一化
    % 计算响应度
    responsivity=N_pdf.*repmat(ComponentProportion,N1,1);  % 响应度responsivity的分子，（X_num,K）的矩阵
    % 归一化响应度（使每行和为1）
    responsivity=responsivity./repmat(sum(responsivity,2),1,K);  % responsivity:在当前模型下第n个观测数据来自第k个分模型的概率，即分模型k对观测数据Xn的响应度
    
    %% 聚类
    [~,Label]=max(responsivity,[],2);
end