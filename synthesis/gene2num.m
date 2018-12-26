function parameter = gene2num(gene)
switch gene
    case 'CDH1-0'
        parameter=1;
    case 'CDH1-70'
        parameter=2;
    case 'CDH1-75'
        parameter=3;
    case 'CDH1-90'
        parameter=4;
    case 'ROCK1-20'
        parameter=5;
    case 'wildtype'
        parameter=6;
end
end