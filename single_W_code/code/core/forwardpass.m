function [t] = forwardpass(t, W1, W2, bw1, hiddenSize)

[~,sl] = size(t.nums);

for i = sl+1:2*sl-1
    kids = t.kids(i,:);
    c1 = t.nodeFeaturesforward(:,kids(1));
    c2 = t.nodeFeaturesforward(:,kids(2));
    p = tanh(W1*c1 + W2*c2 + bw1);
    %p = c1 .* c2;
    %p = (c1+c2)/2;
    %p = vectorsqrt(c1.*c2);
    
    t.nodeFeaturesforward(:,i) = p;
end

end