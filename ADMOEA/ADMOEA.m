function ADMOEA(Global)
% <algorithm> <A>
% Nondominated sorting genetic algorithm III

%------------------------------- Reference --------------------------------
% K. Deb and H. Jain, An evolutionary many-objective optimization algorithm
% using reference-point based non-dominated sorting approach, part I:
% Solving problems with box constraints, IEEE Transactions on Evolutionary
% Computation, 2014, 18(4): 577-601.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2018-2019 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Generate the reference points and random population
    [Z,Global.N] = UniformPoint(Global.N,Global.M);
    MZ = Z - 1;
    Population_PBI   = Global.Initialization();
    Population_NBI = Population_PBI;
    
    Zmin         = min(Population_PBI(all(Population_PBI.cons<=0,2)).objs,[],1);
    Zmax         = max(Population_PBI(all(Population_PBI.cons<=0,2)).objs,[],1);
    Population   = Population_PBI;
    %Population_U = [ Population_PBI, Population_NBI];
    T = 1;
    %% Optimization
    while Global.NotTermination(Population)
        
        
        MatingPoo  = TournamentSelection(2,Global.N,sum(max(0,Population.cons),2));
        Offspring  = GA(Population(MatingPoo));
        
        Zmin       = min([Zmin;Offspring(all(Offspring.cons<=0,2)).objs],[],1);
%         size(Population_U)
        
        [Population_PBI,T,Zmax] = EnvironmentalSelection([Population_PBI,Offspring,Population_NBI],Global.N,Z,Zmin,MZ,1,T,Zmax);
        [Population_NBI] = EnvironmentalSelection([Population_NBI,Offspring,Population_PBI],Global.N,Z,Zmin,MZ,2,T,Zmax);
       
       
        if(T == 1)
            Population = Population_PBI;
        else
            Population = Population_NBI;
        end
    end
end