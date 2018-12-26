    function out = imageTrim(network, k)
[r c w] = size(network);
switch k
    case 1
        d = min([floor(log2(r)),floor(log2(c))]);
        b = 2^d;
        extraR = floor((r-b)/2);
        extraC = floor((c-b)/2);
        out = network((extraR+1):(extraR+b),(extraC+1):(extraC+b),:);
    case 2
        network(1,:,:)=[];
        network(end,:,:)=[];
        network(:,1,:)=[];
        network(:,end,:)=[];
        C=[];
        for jj=1:(c-2)
            if all(network(:,jj,:)==ones(r-2,1,w).*255)
            else
                C=[C jj];
            end
        end
        R=[];
        for ii=1:(r-2)
            if all(network(ii,:,:)==ones(1,c-2,w).*255)
            else
                R=[R ii];
            end
        end
        out = network(R,C,:);
        % We need images to be at least 32 by 32 to have depth 5 quadtrees:
        if length(R)<32
            out=[out;uint8(ones(32-length(R),length(C),w).*255)];
        end
        if length(C)<32
            out=[out,uint8(ones(size(out,1),32-length(C),w).*255)];
        end
        out=squareMean(out);
end
end