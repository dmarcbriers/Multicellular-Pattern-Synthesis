% usage: 
% SimulationScores.m "folder_path" "weka_file" "output_file"
%
% computes the TSSL robustness scores for all the images in a folder,
% stores the results in a file with a user-defined name, and saves the
% scores plot as an image file called scores.png
%
% positional arguments:
% folder_path: the path of the folder that contains simulation images
% weka_file: the name of the Weka classifier corresponding to a pattern
% output_file: the name of a file specified to store the results
%
% example:
% octave SimulationScores.m "../model/simulations/wildtype_-144_CDH1-90_8.0609_15_2" "BullseyeRules1.txt" "scores.txt" 


clear
arg_list = argv ();
if nargin < 3
    error('folder_path, weka_file and output_file must be specified');
elseif nargin > 3
    error('too many positional arguments');
end
folderPath = arg_list{1};
weka_file = arg_list{2};
formula = getFormulaVariance(weka_file,1024);
output_file = arg_list{3}
[robustness, m]=findRobustness(folderPath,formula,1,output_file, true)