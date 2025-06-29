function label = RefSelection(Problem, Population, Wb)
    ArcObj = Population.objs;
    
    % DDS
    % [DDrank, ~] = ImDD(ArcObj, Wb);
    DDrank = ImDD(ArcObj, Wb);
    DDrank = DDrank-1;
 
    % DDNDS
    [FrontNo, ~] = BiasSort(ArcObj);
    BSrank = FrontNo-1;

    % NDS
    % [FrontNoNDS,~] = NDSort(ArcObj, inf);
    % FNDS = FrontNoNDS - 1; 
    
    alpha = Problem.FE / (Problem.maxFE - Problem.N);
    rank = (1 - alpha) *DDrank .* (alpha * BSrank);
    % rank = DDrank .* BSrank;
    % rank = BSrank;
    % rank = DDrank;
    % rank = FNDS;
    
    label = (rank==0);

end