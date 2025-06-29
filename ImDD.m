function rank = ImDD(obj, V)
% function [rank, p] = ImDD(obj, V)
    % find extreme solutions
    indexE = [];
    for i = 1 : size(V,1)
         y = TCH(obj, V(i,:));
         [~, indexe] = sort(y, "ascend");
         indexE = [indexE, indexe(1)];
    end
    ExtremeObj = obj(indexE,:);

    Q = ExtremeObj;
    P = obj(setdiff(1:size(obj,1), indexE),:);
    rank = zeros(1, size(obj,1));
    rank(indexE) = 1;

    % the first pivot
    Zidl = min(Q, [], 1);

    for j = 1 : size(P,1)
        % get pivot
        pivotV = Zidl ./ sum(Zidl);

        % find neighbor solutions
        Angle = acos(1-pdist2(P, pivotV, 'cosine'));
        [~, idx1] = sort(Angle, 'ascend');
        ns = ceil(size(P, 1) * 0.1);
        if ns <= 0
            break
        end
        NP = P(idx1(1:ns), :);

        % caculate each NP's TCH values 
        y = TCH(NP, pivotV);
        [~, idx2] = sort(y, 'ascend');
        
        label = find(all(obj == NP(idx2(1),:), 2));
        rank(label) = j + 1;
        Q = [Q; NP(idx2(1),:)];

        label2 = find(all(P ~= NP(idx2(1),:), 2));
        P = P(label2, :);
        
        % the next pivot
        Zidl = min(P, [], 1);
        
        % 找到第一个正式排序的个体
        % if j == 1
        %     p = find(all(P == NP(idx2(1),:), 2));
        % end
    end
end

function y = TCH(obj, Vi)
    Zl = min(obj, [], 1); 
    y = max(abs(obj - repmat(Zl, size(obj,1), 1)) .* Vi, [], 2);
end