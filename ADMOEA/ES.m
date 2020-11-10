function [Population,T] = ES(Population,N,Z,Zmin,MZ,type,T)
% The environmental selection of NSGA-III

%------------------------------- Copyright --------------------------------
% Copyright (c) 2018-2019 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    if isempty(Zmin)
        Zmin = ones(1,size(Z,2));
    end

    %% Non-dominated sorting
%      Population = unique(Population);
    [FrontNo,MaxFNo] = NDSort(Population.objs,Population.cons,N);
    Zmax = max([Population(FrontNo==1).objs;ones(1,size(Population.objs,2))*0.000001])
%     NDP = find(sum(Population.objs>Zmax,2)>0);
%     FrontNo(NDP) = MaxFNo+1;
    Next = FrontNo<=MaxFNo
     
    %% Select the solutions in the last front
%     Last   = find(FrontNo==MaxFNo);
    %Last   = Last(find(sum(Population(Last).objs>Zmax,2)<0));
    if type == 1
        Choose = LastSelectionPBI(Population(Next).objs,FrontNo(Next),N,Z,Zmin,Zmax);
    else
        Choose = LastSelectionNBI(Population(Next).objs,FrontNo(Next),N,Z,Zmin,Zmax,MZ);
    end
%     Next(Last(Choose)) = true;
    % Population for next generation
    Population = Population(Choose);
   
    if(sum(sum((Population.objs-Zmin)./Zmax,2)-1>=0)>N/2)
        T = 1;
    else
        T = 2;
    end
    
end

function Choose = LastSelectionPBI(PopObj,FrontNo,K,Z,Zmin,Zmax)
% Select part of the solutions in the last front

    PopObj =  PopObj - repmat(Zmin,size(PopObj,1),1);
    [N,M]  = size(PopObj);
%     N1     = size(PopObj1,1);
%     N2     = size(PopObj2,1);
    NZ     = size(Z,1);

    %% Normalization
    % Detect the extreme points
    % Normalization
    PopObj = PopObj./Zmax;
    
    %% Associate each solution with one reference point
    % Calculate the distance of each solution to each reference vector
    Cosine   = 1 - pdist2(PopObj,Z,'cosine');
    
    Distance = repmat(sqrt(sum(PopObj.^2,2)),1,NZ).*sqrt(1-Cosine.^2);
    
    % Associate each solution with its nearest reference point
    [d,pi] = min(Distance',[],1);

    %% Calculate the number of associated solutions except for the last front of each reference point
    rho = zeros(1,NZ);
    
    %% Environmental selection
    Choose  = false(1,N);
    Zchoose = true(1,NZ);
    % Select K solutions one by one
    while sum(Choose) < K
        % Select the least crowded reference point
        Temp = find(Zchoose);
        Jmin = find(rho(Temp)==min(rho(Temp)));

        j    = Temp(Jmin(randi(length(Jmin))));
        I    = find(Choose==0 & pi(1:end)==j);
        % Then select one solution associated with this reference point
        if ~isempty(I)
            F = min(FrontNo(I));
            J = find(FrontNo(I)==F);
            if rho(j) == 0
                F = min(FrontNo(I));
                J = find(FrontNo(I)==F);
                [~,s] = min(d(I(J)));
            else
                s = randi(length(I(J)));
            end
            Choose(I(J(s))) = true;
            rho(j) = rho(j) + 1;
        else
            Zchoose(j) = false;
        end
    end
end

function Choose = LastSelectionNBI(PopObj,FrontNo,K,Z,Zmin,Zmax,MZ)
% Select part of the solutions in the last front

    PopObj = PopObj - repmat(Zmin,size(PopObj,1),1);
    [N,M]  = size(PopObj);
%     N1     = size(PopObj1,1);
%     N2     = size(PopObj2,1);
    NZ     = size(Z,1);

    %% Normalization
   % Detect the extreme points
%     Extreme = zeros(1,M);
%     w       = zeros(M)+1e-6+eye(M);
%     for i = 1 : M
%         [~,Extreme(i)] = min(max(PopObj./repmat(w(i,:),N,1),[],2));
%     end
%     % Calculate the intercepts of the hyperplane constructed by the extreme
%     % points and the axes
%     Hyperplane = PopObj(Extreme,:)\ones(M,1);
%     a = 1./Hyperplane;
%     if any(isnan(a))
%         a = max(PopObj,[],1)';
%     end
    % Normalization
    PopObj = PopObj./Zmax;
    
    %% Associate each solution with one reference point
    % Calculate the distance of each solution to each reference vector
    Cosine   = ones(N,NZ);
    Length   = ones(N,NZ);
    
    for i = 1:NZ  
        Cosine(:,i)   = 1 - pdist2(PopObj-repmat(MZ(i,:),N,1),(Z(i,:)-MZ(i,:)),'cosine');
        Length(:,i)   = sqrt(sum((PopObj-repmat(MZ(i,:),N,1))-repmat(Z(i,:)-MZ(i,:),N,1).^2,2));
    end
    Distance = Length.*sqrt(1-Cosine.^2);
    % Associate each solution with its nearest reference point
    [d,pi] = min(Distance',[],1);

    %% Calculate the number of associated solutions except for the last front of each reference point
    rho = zeros(1,NZ);
    
    %% Environmental selection
    Choose  = false(1,N);
    Zchoose = true(1,NZ);
    % Select K solutions one by one
    while sum(Choose) < K
        % Select the least crowded reference point
        Temp = find(Zchoose);
        Jmin = find(rho(Temp)==min(rho(Temp)));

        j    = Temp(Jmin(randi(length(Jmin))));
        I    = find(Choose==0 & pi(1:end)==j);
        % Then select one solution associated with this reference point
        if ~isempty(I)
            F = min(FrontNo(I));
            J = find(FrontNo(I)==F);
            if rho(j) == 0
                F = min(FrontNo(I));
                J = find(FrontNo(I)==F);
                [~,s] = min(d(I(J)));
            else
                s = randi(length(I(J)));
            end
            Choose(I(J(s))) = true;
            rho(j) = rho(j) + 1;
        else
            Zchoose(j) = false;
        end
    end
end