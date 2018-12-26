function out=make32(network)
[r c w]=size(network);
if r==32 && c==32
    out=network;
else
    d=log2(r);
    D=2^(d-5);
    temp1=[];
    for ii=0:31
        temp=network((D*ii+1):(D*(ii+1)),:,:);
        temp1=uint8([temp1;mean(temp)]);
    end
    out=[];
    for jj=0:31
        temp=temp1(:,(D*jj+1):(D*(jj+1)),:);
        out=uint8([out,mean(temp,2)]);
    end
end
end