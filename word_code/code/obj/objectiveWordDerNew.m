function [cost] = objectiveWordDerNew(hiddenSize, d, pairs, words, We_old, We,lambda, lambdawords, margin, toprint)

if(nargin < 10)
    toprint = 0;
end

%assume trees are propagated forward
% = reshape(We,25,27);
[a b] = size(We);
if (a ==1 || b == 1) 
   n = numel(We);
   v = n/hiddenSize;
   We = reshape(We,hiddenSize,v);
end

total = 0;
for ii=1:1:length(d)
    %fprintf('Computing cost for tree %d\n',ii);    
    t1 = d{ii}(1);
    t2 = d{ii}(2);
    p1 = pairs{ii}(1);
    p2 = pairs{ii}(2);
    
    g1 = We(:,t1);
    g2 = We(:,t2);
    v1 = We(:,p1);
    v2 = We(:,p2);
    
    d1 = margin - sum(g1.*g2) + sum(v1.*g1);
    d2 = margin - sum(g1.*g2) + sum(v2.*g2);

    if(toprint)
       % [sum(g1.*g2) sum(v1.*g1) sum(v2.*g2) d1 d2 words(t1.nums) words(t2.nums) words(p1.nums) words(p2.nums)]
    end

    if(d1 < 0)
        d1 = 0;
    end
    
    if(d2 < 0)
        d2 = 0;
    end
    
    %[d1 d2 sum(g1.*g2) sum(v1.*g1) sum(v2.*g2)]
    
    total = total + d1 + d2;
end

total = total / length(d);
total3 = sum(sum((We_old-We).^2)) * lambdawords / 2;
cost = total+total3;

end