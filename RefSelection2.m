function label = RefSelection2(Problem, Population, c)
    Objs = Population.objs;
    [Label, mu] = GMM(Problem, Objs, c);
    
    label = [];
    % find neighbor solutions
    for i = 1 : c
        obji = Objs(Label==i,:);
        mui = mu(i,:);
        y = TCH(obji, mui);
        [~, idx] = sort(y,'ascend');
        l = find(all(Objs == obji(idx(1),:), 2));
        label = [label, l];
    end

end

function y = TCH(obj, Vi)
    Zl = min(obj, [], 1); 
    y = max(abs(obj - repmat(Zl, size(obj,1), 1)) .* Vi, [], 2);
end