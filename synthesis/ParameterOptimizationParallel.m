% usage: matlab ParameterOptimizationParallel.m
%
%runs particle swarm optimization to find parameters in the cellular potts
% model that result in the best TSSL score
%
% variables in lines 9 to 15 may be modified 

clear
particles = 30; % Total number of particles for PSO
parameters = 5; % Total number of paramters that we want to synthesize
bounds = [1 6.99;-144 96;1 6.99;-144 96;10 50]; % Bounds for all parameters
maxIterations = 30; % Maximum number of PSO iterations
weka_file = 'BullseyeRules1.txt'; % File containing Weka rules
formula = getFormulaVariance(weka_file,1024);
output_file = 'OptimizationResultsParallel.txt'; % Name for output file
parpool()
tic
[ optimised_parameters ] = Particle_Swarm_Optimization_Parallel ...
    (particles, parameters, bounds, 'averageScore', 'max', 2, 2.5, 1.5, 0.4, 0.9, maxIterations,formula,output_file);
toc
delete(gcp('nocreate'));