function [Trees, vectors] = getTrees3(parseTrees, W1, W2, bw1, We_orig, hiddenSize, f)

vectors = {};
Trees = {};

for ii = 1:length(parseTrees)  
    words_indexed = cell2mat(parseTrees(ii).Num);
    words_embedded = We_orig(:, words_indexed);
    words_embedded = words_embedded(1:hiddenSize,:);
    
    [~, sl] = size(words_embedded);
    
    Tree = tree2;
    Tree.pp = zeros((2*sl-1),1);
    Tree.nodeScores = zeros(2*sl-1,1);
    Tree.nodeNames = 1:(2*sl-1);
    Tree.kids = zeros(2*sl-1,2);
    
    Tree.nodeFeaturesforward = zeros(hiddenSize, 2*sl-1);
    Tree.nodeFeaturesforward(:,1:sl) = words_embedded;

    for i = sl+1:2*sl-1
        arr = cell2mat(parseTrees(ii).Kids);
        kids = arr(i,:);
        c1 = Tree.nodeFeaturesforward(:,kids(1));
        c2 = Tree.nodeFeaturesforward(:,kids(2));
        %p = tanh(W1*c1 + W2*c2 + bw1);       
        p = f(W1,W2,bw1,c1,c2);
        Tree.nodeFeaturesforward(:,i) = p;
    end

    Tree.pp = cell2mat(parseTrees(ii).Tree);
    Tree.kids = cell2mat(parseTrees(ii).Kids);
    Tree.nums = cell2mat(parseTrees(ii).Num);
    
    vectors{end+1} = Tree.nodeFeaturesforward(:,end)';
    Trees{end+1} = Tree;
end

end