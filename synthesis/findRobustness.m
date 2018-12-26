%Computes TSSL robustness score for all images in a folder
%imageFolder: name of folder containing the images (string)
%wekaFile: name of file containing WEKA rules (string)
%color: the color channel that is used to build quadtrees; color=1 for red,
%color=2 for green, color =3 for blue
%outputFile: output file name in which robustness scores for all images are
%stored (string)
%makePlot: either true or false. if true, plots the robustness scores
%robustness: a vector containing all the robustness scores
%m: the maximum robustness value

function [robustness, m]=findRobustness(imageFolder,formula,color,outputFile, makePlot)

if ~isa(imageFolder, 'char')
    error('imageFolder must be a string')
elseif ~ismember(color, [1,2,3])
    error('color must be 1 or 2 or 3')
elseif ~isa(makePlot, 'logical')
    error('makePlot must be logical true or false')
end
fclose all;
if isa(outputFile, 'char')
    fid=fopen(outputFile,'w');
end
listing=dir(imageFolder);
len=length(listing);
ind=[];
for ii=1:len
    names=listing(ii).name;
    if length(names)>=5 && all(names((end-2):end)=='png')
        ind=[ind ii];
    end
end
listing=listing(ind);
len=length(listing);
robustness = zeros(1,len);
for ii=1:len
    names=listing(ii).name;
    fileName=fullfile(imageFolder,names);
    data=imread(fileName);
    data=imageTrim(data,2);
    data=make32(data);
    data=double(data(:,:,color));
    data=data./max(max(data));
    d=log2(size(data,1));
    [quadtree{1}, quadtree{2}]=calculateQuadtree(data);
    robustness(1,ii) = modelCheckPattern(formula, quadtree, 'P ');
    if isa(outputFile, 'char')
        fprintf(fid,names);
        fprintf(fid,': ');
        fprintf(fid,num2str(robustness(1,ii)));
        fprintf(fid,'\n');
    end
end
m=max(robustness,[],2);
if isa(outputFile, 'char')
    fprintf(fid, 'The maximum robustness is ');
    fprintf(fid, num2str(m));
    fprintf(fid, '\n');
    fclose(fid);
end
if makePlot==true
    clf();
    plot(robustness);
    saveas(1,'Scores.png');
end
