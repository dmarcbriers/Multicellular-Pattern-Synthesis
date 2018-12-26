% Computes the quadree
function [tree, varTree] = calculateQuadtree(network)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here



d = 1; % FIX THIS
network_size = size(network,1)/d;

ts = [log2(network_size)+1, (network_size)^2];



tree = zeros(ts(1), ts(2));
tree(1,:) =  mean(mean(network)); 


if(nargout > 1)
    varTree = zeros(ts(1), ts(2));
    varTree(1,:) = sum(sum(network.*network - tree(1,1)*tree(1,1)))/(4*network_size*network_size); 
    % divide by 4 for normalization
    % variance of n equally likely values
end



if (network_size == 1)
    return;
end


nsO = size(network,1);

ns = nsO/2;

CI = [1,1; 1,ns+1; ns+1,ns+1; ns+1, 1]; 


S = (network_size)^2/4;
 
for i=1:4
    if(nargout > 1)
        [ tree(2:end, 1+S*(i-1):S*i), varTree(2:end, 1+S*(i-1):S*i) ] = calculateQuadtree(network(CI(i,1):CI(i,1)+ns-1, CI(i,2):CI(i,2)+ns-1));
    else
        tree(2:end, 1+S*(i-1):S*i) = calculateQuadtree(network(CI(i,1):CI(i,1)+ns-1, CI(i,2):CI(i,2)+ns-1));
    end 
end


return




 