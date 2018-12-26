% usage: 
% ParameterOptimization.m "weka_file" "output_file"
%
% runs particle swarm optimization to find parameters in the cellular potts
% model that result in the best TSSL score
%
% positional arguments:
% weka_file: the name of the Weka classifier corresponding to a pattern
% output_file: the name of a file specified to store the results
%
% example:
% octave ParameterOptimization.m "BullseyeRules1.txt" "OptimizationResults.txt" 

clear
arg_list = argv ();
if nargin < 2
    error('weka_file and output_file must be specified');
elseif nargin > 2
    error('too many positional arguments');
end
particles = 30; % Total number of particles for PSO
parameters = 5; % Total number of paramters that we want to synthesize
bounds = [1 6.99;-144 96;1 6.99;-144 96;10 50]; % Bounds for all parameters
maxIterations = 30; % Maximum number of PSO iterations 
weka_file = arg_list{1}; % File containing Weka rules
formula = getFormulaVariance(weka_file,1024);
output_file = arg_list{2};  % Name for output file
tic
[ optimised_parameters ] = Particle_Swarm_Optimization_Parallel ...
    (particles, parameters, bounds, 'averageScore', 'max', 2, 2.5, 1.5, 0.4, 0.9, maxIterations,formula,output_file);
toc

