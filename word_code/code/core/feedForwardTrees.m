function [data] = feedForwardTrees(data, theta, hiddenSize, We)

W1 = reshape(theta(1:hiddenSize*hiddenSize),hiddenSize,hiddenSize);
W2 = reshape(theta(hiddenSize*hiddenSize+1:2*hiddenSize*hiddenSize),hiddenSize,hiddenSize);
bw1 = reshape(theta(2*hiddenSize*hiddenSize+1:2*hiddenSize*hiddenSize+hiddenSize),hiddenSize,1);

for ii=1:1:length(data)
    %fprintf('Computing cost for tree %d\n',ii);
    currTreep1 = data{ii}{1};
    currTreep2 = data{ii}{2};
    treesp1 = forwardpassWordDer(currTreep1, W1, W2, bw1, hiddenSize, We);
    treesp2 = forwardpassWordDer(currTreep2, W1, W2, bw1, hiddenSize, We);
    data{ii}{1} = treesp1;
    data{ii}{2} = treesp2;
end

end