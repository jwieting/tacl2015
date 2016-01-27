function [pairs] = getPairs(sample,words,We)

global wordmap;
global params;

[n,~]=size(We);
mat = zeros(n,2*length(sample));

for i=1:1:length(sample)
    mat(:,2*i-1) = We(:,sample{i}(1));
    mat(:,2*i) = We(:,sample{i}(2));
end


%pick closest tree as negative.
length(sample);
pairs = {};
for i=1:1:length(sample)
    s = sample{i};
    x1 = repmat(We(:,sample{i}(1)),[1,length(sample)*2]);
    dp1 = sum((x1.*mat));
    
    x2 = repmat(We(:,sample{i}(2)),[1,length(sample)*2]);
    dp2 = sum((x2.*mat));
    gg = dp1(2*i);
    
    t1 = s(1);
    t2 = s(2);
    
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
                currt = sample{round(j/2)}(1);
            else
                currt = sample{j/2}(2);
            end
            
            if(comparet(currt,t1) > 0)
                continue;
            end

            if(params.constraints && comparet2(wordmap,words,currt,t1) > 0)
                continue;
            end

            mintree1 = currt;
            mintree1score = dp1(j);
        end
        
        if(dp2(j) > mintree2score)
            currt = {};
            if(mod(j,2)==1)
                currt = sample{round(j/2)}(1);
            else
                currt = sample{j/2}(2);
            end          
            
            if(comparet(currt,t2) > 0)
                continue;
            end

            if(params.constraints && comparet2(wordmap,words,currt,t2) > 0)
                continue;
            end

            mintree2score = dp2(j);
            mintree2 = currt;
        end
    end
    %[words(t1) words(mintree1)]
    %[words(t2) words(mintree2)]
    pairs{end+1} = [mintree1 mintree2];
end

end

function [bool] = comparet(t1,t2)
    if(t1==t2)
        bool = 1;
        %disp('skipping');
    else
        bool = 0;
    end
end

function [bool] = comparet2(wordmap,words,t1,t2)
    bool=0;
    if(~isKey(wordmap,words{t1}))
        return;
    end
    l=wordmap(words{t1});
    w1=words(t2);
    for i=1:1:length(l)
        ww=l{i};
        if(strcmp(ww,w1))
            bool=1;
        end
    end
end