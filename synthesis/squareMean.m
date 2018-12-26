function out=squareMean(network, d)
[r c w] = size(network);
dr=floor(log2(r));
br=2^dr;
dc=floor(log2(c));
bc=2^dc;
if nargin==1
    d = min([floor(log2(r)),floor(log2(c))]);
end
b=2^d;
ii=1;
while 1
    if r==br
        break
    end
    temp=network(ii:(ii+1),:,:);
    if ii==1
        network=vertcat(mean(temp),network(((ii+2):end),:,:));
    else
        network=vertcat(network(1:(ii-1),:,:),mean(temp), ...
            network((ii+2):end,:,:));
    end
    [r c w] = size(network);
    if r==br;
        break
    end
    temp=network((end-ii):(end-ii+1),:,:);
    if ii==1
        network=vertcat(network(1:(end-ii-1),:,:),mean(temp));
    else
        network=vertcat(network(1:(end-ii-1),:,:),mean(temp), ...
            network((end-ii+2):end,:,:));
    end
    [r c w] = size(network);
    if r==br;
        break
    end
    ii=ii+1;
end
jj=1;
while 1
    if c==bc
        break
    end
    temp=network(:,jj:(jj+1),:);
    if jj==1
        network=horzcat(mean(temp,2),network(:,(jj+2):end,:));
    else
        network=horzcat(network(:,1:(jj-1),:),mean(temp,2), ...
            network(:,(jj+2):end,:));
    end
    [r c w] = size(network);
    if c==bc;
        break
    end
    temp=network(:,(end-jj):(end-jj+1),:);
    if jj==1
        network=horzcat(network(:,1:(end-jj-1),:),mean(temp,2));
    else
        network=horzcat(network(:,1:(end-jj-1),:),mean(temp,2), ...
            network(:,(end-jj+2):end,:));
    end
    [r c w] = size(network);
    if c==bc;
        break
    end
    jj=jj+1;
end
while r~=b
    temp=[];
    for ii=1:r/2
        temp=uint8([temp;mean(network(ii*2-1:ii*2,:,:))]);
    end
    network=temp;
    [r c w] = size(network);
end
while c~=b
    temp=[];
    for jj=1:c/2
        temp=uint8([temp,mean(network(:,jj*2-1:jj,:),2)]);
    end
    network=temp;
    [r c w] = size(network);
end
out=network;
end