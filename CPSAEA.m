classdef CPSAEA < ALGORITHM
% <multi/many> <real/integer> <expensive>

%------------------------------- Copyright --------------------------------
% Copyright (c) 2021 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            [SubN,k,c] = Algorithm.ParameterSet(ceil(Problem.N/2), 5, 3 * Problem.M);

            %% Generate random population
            PopDec     = UniformPoint(Problem.N, Problem.D, 'Latin');
            Population = Problem.Evaluation(repmat(Problem.upper - Problem.lower, Problem.N, 1) .* PopDec + repmat(Problem.lower, Problem.N, 1));
            [W, c]  = UniformPoint(c,Problem.M);

            % reference of HV
            Reference    = max(Population.objs,[],1); 
            
            % MaxHV
            maxhv = 0;
            s = 0;

            %% Optimization
            while Algorithm.NotTerminated(Population)
                
                % 选择RefPop
                % RefPop = EnvironmentalSelection(Population, c);

                ind = HV(Population.objs,Reference);
                if ind > maxhv
                    RefPop = EnvironmentalSelection(Population, c);
                    maxhv = ind;
                else
                    RefPop = Population(RefSelection(Population, W));
                    s = 1;
                end

                % if Problem.FE/(Problem.maxFE-Problem.N) < 0.5
                %     RefPop = Population(RefSelection3(Population, W));
                % else
                %     RefPop = EnvironmentalSelection(Population, c);
                % end
                
                % RefPop = Population(RefSelection(Population, W));
                % RefPop = EnvironmentalSelection(Population, c);
                
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
                % if s == 1
                %     W = UpdateV(Problem, Population, W, c);
                %     s=0;
                % end

                
                % Update reference of HV
                Reference    = max(Population.objs,[],1);
            end

            
        end
    end
end