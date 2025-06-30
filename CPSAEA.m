classdef CPSAEA < ALGORITHM
% <multi/many> <real/integer> <expensive>

%------------------------------- Reference --------------------------------
% Q. Zhang, W. Liu, E. Tsang, and B. Virginas, Expensive multiobjective
% optimization by MOEA/D with Gaussian process model, IEEE Transactions on
% Evolutionary Computation, 2010, 14(3): 456-474.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2021 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Cheng He

    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            [SubN,k,c] = Algorithm.ParameterSet(ceil(Problem.N/2), 5, 2 * Problem.M);

            %% Generate random population
            PopDec     = UniformPoint(Problem.N, Problem.D, 'Latin');
            Population = Problem.Evaluation(repmat(Problem.upper - Problem.lower, Problem.N, 1) .* PopDec + repmat(Problem.lower, Problem.N, 1));
            [W, SubN]  = UniformPoint(SubN,Problem.M);

            Wb = eye(Problem.M);
            Wb(Wb==0) = 1e-6;

            %% Optimization
            while Algorithm.NotTerminated(Population)
                % 选择RefPop
                % RefPop = EnvironmentalSelection(Population, c);
                RefPop = Population(RefSelection3(Population, W));
                
                % Reconstruct Problem
                Off = RCPSAEA(Problem, Population, RefPop, SubN);


                % Environmental selection
                % candidates = EnvironmentalSelection(Off, Wb, k);
                % k = min(k, length(Off));
                [candidates,~,~] = EnvironmentalSelection(Off, min(k, length(Off)));

                % Offspring = Problem.Evaluation(Off.decs);
                Offspring = Problem.Evaluation(candidates.decs);
                Population = [Population,Offspring];

                % update weight vector
                W = UpdateV(Problem, Population, W, c);

                
                
            end

            
        end
    end
end