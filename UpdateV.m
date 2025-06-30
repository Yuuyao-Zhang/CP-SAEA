function W = UpdateV(Problem, Population, W, c) 
    [~, partition] = max(1-pdist2(Population.objs, W, 'cosine'),[],2);
    uselessV = setdiff(1:size(W,1), unique(partition), 'stable');

    [~, mu] = GMM(Problem, Population.objs, c);
    c = min(length(uselessV), c);
    r = randperm(length(uselessV), c);

    for i = 1 : c
        ri = uselessV(r(i));
        mui = mu(i,:);
        W(ri,:) = mui;
    end
end