function PopObj = RBFPredictor(Problem, PopDec, rbfCell)
    M = Problem.M;
    PopObj = [];
    for i = 1: M
        rbfi = rbfCell{i};
        obj = sim(rbfi, PopDec')';
        PopObj = [PopObj,obj];
    end
end