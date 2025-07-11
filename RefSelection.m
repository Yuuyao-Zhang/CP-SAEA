function label = RefSelection(Population, W)
    Objs = Population.objs;
    [~, partition] = max(1-pdist2(Objs, W, 'cosine'),[],2);
    label = [];

    % find neighbor solutions
    for i = unique(partition)'
        obji = Objs(partition==i,:);
        wi = W(i,:);
        y = TCH(obji, wi);
        [~, idx] = sort(y,'ascend');
        l = find(all(Objs == obji(idx(1),:), 2));
        try
            label = [label, l];
        catch
            disp(label);
        end
    end

end

function y = TCH(obj, Vi)
    Zl = min(obj, [], 1); 
    y = max(abs(obj - repmat(Zl, size(obj,1), 1)) .* Vi, [], 2);
end
