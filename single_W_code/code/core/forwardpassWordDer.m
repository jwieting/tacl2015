function [t] = forwardpassWordDer(t, W1, W2, bw1, hiddenSize, We_orig)

[~,sl] = size(t.nums);

words_indexed = t.nums;
words_embedded = We_orig(:, words_indexed);
t.nodeFeaturesforward = zeros(hiddenSize, 2*sl-1);
t.nodeFeaturesforward(:,1:sl) = words_embedded;

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