function Offspring = RCPSAEA(Problem, Population, RefPop, SubN)
    
    %% Calculate the directions
	lower     = Problem.lower;
    upper     = Problem.upper;
    wD = size(RefPop,2);
    % reference of HV
    Reference    = max(Population.objs,[],1); 

    NorOpt = sum((RefPop.decs-repmat(Problem.lower,length(RefPop),1)).^2,2).^(0.5); % 计算RefDec与上下界的距离
    RefPopFix    = (RefPop.decs-repmat(lower,wD,1))./repmat(NorOpt,1,Problem.D); % 对RefDec先进行偏移矫正，再进行归一化
	wmax      = sum((upper-lower).^2)^(0.5)*0.5; % 界限值

    %% Create RBFN
    rbfCell = cell(1,Problem.M);
    PopDec = Population.decs;
    PopObj = Population.objs;

    for i = 1:Problem.M
        rbfCell{i} = newrb(PopDec', PopObj(:,i)', 0.01);
    end

    %% Optimize reconstruct problem

    % Initlization
    w0 = rand(SubN, length(RefPop)).* wmax; 
    [fitness, ~] = FitCal(Problem, rbfCell, w0, RefPopFix, Reference);

    % Construct regression model (w0 -> fitness (HV))
    [krigingModel, ~] = dacefit(w0, fitness,'regpoly0','corrgauss',1*ones(1,length(RefPop)), 0.001*ones(1,length(RefPop)), wmax*ones(1,length(RefPop)));
    
    % Find the current optimal solution
    [~, idx] = sort(fitness, 'ascend');
    bestInd.decs = w0(idx(1),:);
    bestInd.objs = fitness(idx(1),:);
    status = 0; % status of reference point update
    
    % Generate new individuals
    itermax = 5;
    for i = 1 : itermax
        for j = 1 : SubN
            dec = w0;
            A = randperm(SubN);
            A(A==j) = [];
            a = A(1); b = A(2); c = A(3);
            o = DE(dec(a,:), dec(b,:), dec(c,:), zeros(1,size(dec,2)), repmat(wmax,1,size(dec,2)));
            
            % Evaluate new individual
            for it = 1 : size(o, 1)
                [oy, omse] = predictor(o(it,:), krigingModel);
                
                % Update best individual of reconstruct problem
                if oy < bestInd.objs
                    % Confidence verification
                    [by, bmse] = predictor(bestInd.decs, krigingModel);
                    dis = (oy + 3 * sqrt(omse)) - (by - 3 * sqrt(bmse));
                    if dis > 0
                        bestInd.decs = o(it,:);
                        bestInd.objs = oy;
                        status = 1;
                    end
                end
                
                % Update reference point
                % if status == 1
                %     status = 0;
                %     [~, Offspring] = FitCal(Problem, rbfCell, bestInd.decs, RefPopFix, Reference);
                %     pop = [RefPop,Offspring];
                %     [RefPop,~,~] = EnvironmentalSelection(pop,length(RefPop));
                % end
            end
            
            % Transform to origin problem
            [~, Offspring] = FitCal(Problem, rbfCell, bestInd.decs, RefPopFix, Reference);
        end
    end


    % Data augmentation
    % OffDec = Offspring.decs;
    % OffObj = Offspring.objs;
    % [FrontNo1,~] = NDSort(OffObj,inf);
    % trainx1 = OffDec(FrontNo1==1,:);
    % 
    % GANDec = GAN(trainx1, Problem);
    % GANObj = RBFPredictor(Problem, GANDec, rbfCell);
    % [FrontNo2,~] = NDSort(GANObj,inf);
    % 
    % GANoff = SOLUTION(GANDec,GANObj,zeros(size(GANDec,1),1));
    % Offspring = [Offspring(FrontNo1==1), GANoff(FrontNo2==1)];
    
    % No data augmentation
    [FrontNo1,~] = NDSort(Offspring.objs,inf);
    Offspring = Offspring(FrontNo1==1);
    

end

function [fitness, offspring] = FitCal(Problem, rbfCell, w0, RefPopFix, Reference)
    [SubN,WD] = size(w0);
    fitness   	= zeros(SubN,1);
    offspring  = [];
    
    for i = 1 : SubN 
        PopDec    = repmat(w0(i,:)',1,Problem.D).*RefPopFix + repmat(Problem.lower,WD,1);  % Problem transform back
        PopObj    = RBFPredictor(Problem, PopDec, rbfCell);

        % 封装成solution类
        offPop = SOLUTION(PopDec,PopObj,zeros(size(PopDec,1),1));
        offspring = [offspring, offPop];
        fitness(i)    = -HV(offPop.objs,Reference); % 转换成最小化问题
    end
    
end