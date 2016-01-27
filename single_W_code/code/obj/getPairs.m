function [pairs] = getPairs(sample,words)

if(length(sample) > 0)
    [n,~] = size(sample{1}{1}.nodeFeaturesforward);
end

mat = zeros(n,2*length(sample));

for i=1:1:length(sample)
    mat(:,2*i-1) = sample{i}{1}.nodeFeaturesforward(:,end);
    mat(:,2*i) = sample{i}{2}.nodeFeaturesforward(:,end);
    %words(sample{i}{1}.nums);
    %words(sample{i}{2}.nums);
end


%pick closest tree as negative.
length(sample);
pairs = {};
for i=1:1:length(sample)
    s = sample{i};
    x1 = repmat(s{1}.nodeFeaturesforward(:,end),[1,length(sample)*2]);
    dp1 = sum((x1.*mat));
    
    x2 = repmat(s{2}.nodeFeaturesforward(:,end),[1,length(sample)*2]);
    dp2 = sum((x2.*mat));
    gg = dp1(2*i);
    
    t1 = s{1};
    t2 = s{2};
    
    mintree1 = {};
    mintree2 = {};
    mintree1score = -5;
    mintree2score = -5;
    for j=1:1:length(dp1)
        idxj = round(j/2);
        if(idxj==i)
            continue;
        end
        if(dp1(j) > mintree1score)     
            currt = {};
            if(mod(j,2)==1)
                currt = sample{round(j/2)}{1};
            else
                currt = sample{j/2}{2};
            end
            
            if(comparet(currt,t1) > 0)
                continue;
            end
            
            mintree1 = currt;
            mintree1score = dp1(j);
        end
        
        if(dp2(j) > mintree2score)
            currt = {};
            if(mod(j,2)==1)
                currt = sample{round(j/2)}{1};
            else
                currt = sample{j/2}{2};
            end          
            
            if(comparet(currt,t2) > 0)
                continue;
            end
            
            mintree2score = dp2(j);
            mintree2 = currt;
        end
    end
    %[words(s{1}.nums) words(mintree1.nums)]
    %[words(s{2}.nums) words(mintree2.nums)]
    pairs{end+1} = {mintree1 mintree2};
end

end

function [bool] = comparet(t1,t2)

    if(~(length(t1.nums)==length(t2.nums)))
        bool = 0;
        return;
    end

    if(all(t1.nums == t2.nums))
        bool = 1;
        %disp('skipping');
    else
        bool = 0;
    end
end