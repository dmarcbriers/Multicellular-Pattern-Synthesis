function gene = num2gene(parameter)
switch floor(parameter)
    case 1
        gene='CDH1-0';
    case 2
        gene='CDH1-70';
    case 3
        gene='CDH1-75';
    case 4
        gene='CDH1-90';
    case 5
        gene='ROCK1-20';
    case 6
        gene='wildtype';
end
end