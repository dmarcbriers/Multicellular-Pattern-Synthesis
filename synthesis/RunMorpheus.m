function folderName = RunMorpheus(parameters, tries, simulation_time)
if ~exist('simulation_time','var')
    simulation_time = 96;
end
[STATUS, MSG, MSGID] = copyfile ('../model/*',pwd);
folderName = cell(tries);

type1 = num2gene(parameters(1));
type2 = num2gene(parameters(3));

for ii=1:tries % Total number of simulations for computing the average TSSL score
    
    cmd{ii}=strcat('python3 MorpheusSetup.py "', ...
    type1,'" "', num2str(parameters(2)),'" "', ...
    type2,'" "', num2str(parameters(4)),'" "', ...
    num2str(floor(parameters(5))), '" "', num2str(ii),'"', ...
    ' --simulation_time "', num2str(simulation_time),'"'); %system command that runs simulations 
%system('module load morpheus');
system(cmd{ii});
folderName{ii} = fullfile('simulations',sprintf('%s_%s_%s_%s_%d_%d', ...
   type1,num2str(parameters(2)),type2,num2str(parameters(4)), ...
   floor(parameters(5)),ii)); % Folder contataining simulation images
end
