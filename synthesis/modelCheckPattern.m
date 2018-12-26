
% quantitative model checking:
function result = modelCheckPattern(formula, tree, pattern)





    % first find the indices of the pattern in formula:
    patternFormulas = [];
    for i=1:length(formula)
       
        if(strcmp(formula{i,2},pattern))
            patternFormulas = [patternFormulas, i]; %#ok<AGROW>
        end
    end
    
    

    % do the quantitative check for each of the formulas, but first do the
    % check until patternFormulas(end)
    
    M = patternFormulas(end);
    QA = zeros(M,1); 
    for mi=1:M
        
        QA(mi) = checkConjunction(formula{mi, 1}, tree);

    end

    % over all possible patternFormulas, we will choose the maximal one:
    PR = zeros(length(patternFormulas),1);
    for pi = 1:length(patternFormulas)
       if(patternFormulas(pi) == 1)
           PR(pi) = 100;
       else
        PR(pi) = min(-QA(1:patternFormulas(pi)-1));
       end
       PR(pi) = min(PR(pi), QA(patternFormulas(pi)));
    end
    
    result = max(PR);
    
end


function r  = checkConjunction( C , tree)

    [c, ~] = size(C);
    r = 1; % we will use min.
    
    for ci=1:c
        % color, depth, index
        val = tree{C(ci,1)}(C(ci,2), C(ci,3));
        if(C(ci,5)  == '>')
            d = val - C(ci,4);
        else
            d = C(ci,4) - val;
        end
        d = d/(4^(C(ci,2) - 1)); % discount factor
        r = min(r,d);
    end

end