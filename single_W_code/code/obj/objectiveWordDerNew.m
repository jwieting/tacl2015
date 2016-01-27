function [cost] = objectiveWordDerNew(theta, hiddenSize, d, pairs, words, We_old, We,lambda, lambdawords, margin, toprint)

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

W1 = reshape(theta(1:hiddenSize*hiddenSize),hiddenSize,hiddenSize);
W2 = reshape(theta(hiddenSize*hiddenSize+1:2*hiddenSize*hiddenSize),hiddenSize,hiddenSize);
bw1 = reshape(theta(2*hiddenSize*hiddenSize+1:2*hiddenSize*hiddenSize+hiddenSize),hiddenSize,1);

total = 0;
for ii=1:1:length(d)
    %fprintf('Computing cost for tree %d\n',ii);    
    t1 = d{ii}{1};
    t2 = d{ii}{2};
    p1 = pairs{ii}{1};
    p2 = pairs{ii}{2};
    
    if(isempty(p1) || isempty(p2))
        continue;
    end
    
    t1 = forwardpassWordDer(t1, W1, W2, bw1, hiddenSize, We);
    t2 = forwardpassWordDer(t2, W1, W2, bw1, hiddenSize, We);
    
    p1 = forwardpassWordDer(p1, W1, W2, bw1, hiddenSize, We);
    p2 = forwardpassWordDer(p2, W1, W2, bw1, hiddenSize, We);
    
    g1 = t1.nodeFeaturesforward(:,end);
    g2 = t2.nodeFeaturesforward(:,end);
    
    v1 = p1.nodeFeaturesforward(:,end);
    v2 = p2.nodeFeaturesforward(:,end);
    
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
total2 = sum(sum(W1 .^ 2)) + sum(sum(W2 .^ 2))+ sum(bw1 .^ 2);
total2 = total2 * lambda / 2;
total3 = sum(sum((We_old-We).^2)) * lambdawords / 2;
cost = total+total2+total3;

end