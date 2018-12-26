function out=averageScore(parameters,formula,deleteSimulations)
tries=3;
simulation_time=96;
cmd = cell(1,tries);
m=zeros(1,tries);

folderName = RunMorpheus(parameters, tries, simulation_time);

for ii=1:tries
[~,m(1,ii)]=findRobustness(folderName{ii},formula,1,0,false);
end
out=mean(m);
if deleteSimulations==true
    for ii=1:tries
        rmdir(folderName{ii},'s');
    end
end
end