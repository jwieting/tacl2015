function [grad] = computeGradFastNewObj(d, pairs, theta, hiddenSize, words, We_orig, lambda, margin)

W1 = reshape(theta(1:hiddenSize*hiddenSize),hiddenSize,hiddenSize);
W2 = reshape(theta(hiddenSize*hiddenSize+1:2*hiddenSize*hiddenSize),hiddenSize,hiddenSize);
bw1 = reshape(theta(2*hiddenSize*hiddenSize+1:2*hiddenSize*hiddenSize+hiddenSize),hiddenSize,1);

W1T2 = getT2(W1,hiddenSize);
W2T2 = getT2(W2,hiddenSize);

gradW1 = zeros(size(W1));
gradW2 = zeros(size(W2));
gradbw1 = zeros(size(bw1));
%lambda = .1;

for t=1:1:length(d)
    t1 = d{t}{1};
    t2 = d{t}{2};
    t1 = forwardpassWordDer(t1, W1, W2, bw1, hiddenSize,We_orig);
    t2 = forwardpassWordDer(t2, W1, W2, bw1, hiddenSize,We_orig);
    p1 = pairs{t}{1};
    p2 = pairs{t}{2};
    p1 = forwardpassWordDer(p1, W1, W2, bw1, hiddenSize,We_orig);
    p2 = forwardpassWordDer(p2, W1, W2, bw1, hiddenSize,We_orig);
    
    g1 = t1.nodeFeaturesforward(:,end);
    g2 = t2.nodeFeaturesforward(:,end);
    v1 = p1.nodeFeaturesforward(:,end);
    v2 = p2.nodeFeaturesforward(:,end);
    d1 = margin - sum(g1.*g2) + sum(v1.*g1);
    d2 = margin - sum(g1.*g2) + sum(v2.*g2);
    
    if(d1 > 0 || d2 > 0)
        t1 = getVderW(W1,W2,bw1,t1,hiddenSize,W1T2,W2T2);
        t2 = getVderW(W1,W2,bw1,t2,hiddenSize,W1T2,W2T2);

        t1t2W1 = helperF(t1.nodeFeaturesforward(:,end), t2.dnodeW1{end}, hiddenSize);
        t1t2W1 = t1t2W1 + helperF(t2.nodeFeaturesforward(:,end), t1.dnodeW1{end}, hiddenSize);
        
        t1t2W2 = helperF(t1.nodeFeaturesforward(:,end), t2.dnodeW2{end}, hiddenSize);
        t1t2W2 = t1t2W2 + helperF(t2.nodeFeaturesforward(:,end), t1.dnodeW2{end}, hiddenSize);
        
        t1t2b = helperF2(t1.nodeFeaturesforward(:,end), t2.dnodebw1{end}, hiddenSize);
        t1t2b = t1t2b + helperF2(t2.nodeFeaturesforward(:,end), t1.dnodebw1{end}, hiddenSize);
        
        if(d1 > 0)
            p1 = getVderW(W1,W2,bw1,p1,hiddenSize,W1T2,W2T2);
            gradW1 = gradW1 - t1t2W1 + helperF(t1.nodeFeaturesforward(:,end), p1.dnodeW1{end}, hiddenSize);
            gradW1 = gradW1 + helperF(p1.nodeFeaturesforward(:,end), t1.dnodeW1{end}, hiddenSize);
            gradW2 = gradW2 - t1t2W2 + helperF(t1.nodeFeaturesforward(:,end), p1.dnodeW2{end}, hiddenSize);
            gradW2 = gradW2 + helperF(p1.nodeFeaturesforward(:,end), t1.dnodeW2{end}, hiddenSize);
            gradbw1 = gradbw1 - t1t2b + helperF2(t1.nodeFeaturesforward(:,end), p1.dnodebw1{end}, hiddenSize);
            gradbw1 = gradbw1 + helperF2(p1.nodeFeaturesforward(:,end), t1.dnodebw1{end}, hiddenSize);
        end
        
        if(d2 > 0)
            p2 = getVderW(W1,W2,bw1,p2,hiddenSize,W1T2,W2T2); 
            gradW1 = gradW1 - t1t2W1 + helperF(t2.nodeFeaturesforward(:,end), p2.dnodeW1{end}, hiddenSize);
            gradW1 = gradW1 + helperF(p2.nodeFeaturesforward(:,end), t2.dnodeW1{end}, hiddenSize);
            gradW2 = gradW2 - t1t2W2 + helperF(t2.nodeFeaturesforward(:,end), p2.dnodeW2{end}, hiddenSize);
            gradW2 = gradW2 + helperF(p2.nodeFeaturesforward(:,end), t2.dnodeW2{end}, hiddenSize);
            gradbw1 = gradbw1 - t1t2b + helperF2(t2.nodeFeaturesforward(:,end), p2.dnodebw1{end}, hiddenSize);
            gradbw1 = gradbw1 + helperF2(p2.nodeFeaturesforward(:,end), t2.dnodebw1{end}, hiddenSize);
        end        
    end
end

grad = [gradW1(:); gradW2(:) ; gradbw1(:)] / length(d);
grad = grad + lambda.*theta;

end

function [mat] = helperF(vector, tensor, hiddenSize)

v = reshape(vector,[1,1,hiddenSize]);
v = repmat(v,[hiddenSize hiddenSize 1]);

t = v.*tensor;
mat = sum(t,3);

% mat = zeros(hiddenSize,hiddenSize);
% %mat = repmat(vector,[1 hiddenSize]).*sum(tensor,3);
%     for i=1:1:hiddenSize
%         for j=1:1:hiddenSize
%             der = 0;
%             for k=1:1:hiddenSize
%                 v = vector(k)*tensor(i,j,k);
%                 der = der + v;
%             end
%             mat(i,j) = der;
%         end
%     end
end

function [mat] = helperF2(vector, matrix, hiddenSize)

v = reshape(vector,[1,hiddenSize]);
v = repmat(v,[hiddenSize 1]);

t = v.*matrix;
mat = sum(t,2);
% mat = zeros(hiddenSize,1);
% %mat = repmat(vector,[1 hiddenSize]).*sum(tensor,3);
% for i=1:1:hiddenSize
%     der = 0;
%     for k=1:1:hiddenSize
%         v = vector(k)*matrix(i,k);
%         der = der + v;
%     end
%     mat(i) = der;
% end

end

function [currTree] = getVderW(W1,W2,bw1,currTree,hiddenSize,W1T2,W2T2)

sl = size(currTree.nums,2);
%populate dsigmoids
for s=1:1:2*sl-1
    %take dsigmoid
    currTree.dsigmoid{end+1} = dsigmoid(currTree.nodeFeaturesforward(:,s));
end

%populate node forward tensors
for s=sl+1:1:2*sl-1
    kids = currTree.kids(s,:);
    kidleft = kids(1);
    kidright = kids(2);
    kidleftvector = zeros(hiddenSize,1);
    if(kidleft > 0)
        kidleftvector = currTree.nodeFeaturesforward(:,kidleft);
    end
    
    kidrightvector = zeros(hiddenSize,1);
    if(kidright > 0)
        kidrightvector = currTree.nodeFeaturesforward(:,kidright);
    end
    
    currTree.dnodeW1{end+1}=repmat(currTree.dsigmoid{s}*kidleftvector',[1 1 hiddenSize]);
    currTree.dnodeW2{end+1}=repmat(currTree.dsigmoid{s}*kidrightvector',[1 1 hiddenSize]);
    currTree.dnodebw1{end+1} = repmat(currTree.dsigmoid{s},[1 hiddenSize]).*eye(hiddenSize);
    
    ttw1=currTree.dnodeW1{end};
    ttw2=currTree.dnodeW2{end};
    for i=1:1:hiddenSize
        temp = zeros(hiddenSize,hiddenSize);
        temp(i,:) = ttw1(i,:,i);
        ttw1(:,:,i) = temp;
        
        temp = zeros(hiddenSize,hiddenSize);
        temp(i,:) = ttw2(i,:,i);
        ttw2(:,:,i) = temp;
    end
    currTree.dnodeW1{end} = ttw1;
    currTree.dnodeW2{end} = ttw2;
    
    kids = currTree.kids(s,:);
    kidleft = kids(1);
    kidright = kids(2);
    
    dsig = reshape(currTree.dsigmoid{s},[1 1 hiddenSize]);
    
    if(kidleft > sl && kidright <= sl)
        
        temp = matrixTensorOp2(currTree.dnodeW1{kidleft-sl}, W1T2, hiddenSize);
        currTree.dnodeW1{end}= currTree.dnodeW1{end} + bsxfun(@times, temp, dsig);
        temp = matrixTensorOp2(currTree.dnodeW2{kidleft-sl}, W1T2, hiddenSize);
        currTree.dnodeW2{end}= currTree.dnodeW2{end} + bsxfun(@times, temp, dsig);
        
        temp = matrixMatrixOp(currTree.dnodebw1{kidleft-sl}, W1, hiddenSize);
        currTree.dnodebw1{end}= currTree.dnodebw1{end} + bsxfun(@times, temp, reshape(dsig,1,hiddenSize));
        
    elseif(kidleft <=sl && kidright > sl)
        
        temp = matrixTensorOp2(currTree.dnodeW1{kidright-sl}, W2T2, hiddenSize);
        currTree.dnodeW1{end}= currTree.dnodeW1{end} + bsxfun(@times, temp, dsig);
        temp = matrixTensorOp2(currTree.dnodeW2{kidright-sl}, W2T2, hiddenSize);
        currTree.dnodeW2{end}= currTree.dnodeW2{end} + bsxfun(@times, temp, dsig);
        
        temp = matrixMatrixOp(currTree.dnodebw1{kidright-sl}, W2, hiddenSize);
        currTree.dnodebw1{end}= currTree.dnodebw1{end} + bsxfun(@times, temp, reshape(dsig,1,hiddenSize));
        
    elseif(kidleft > sl && kidright > sl)
        
        temp1 = matrixTensorOp2(currTree.dnodeW1{kidleft-sl}, W1T2, hiddenSize);
        temp2 = matrixTensorOp2(currTree.dnodeW1{kidright-sl}, W2T2, hiddenSize);
        temp = temp1+temp2;
        currTree.dnodeW1{end}= currTree.dnodeW1{end} + bsxfun(@times, temp, dsig);
        
        temp1 = matrixTensorOp2(currTree.dnodeW2{kidleft-sl}, W1T2, hiddenSize);
        temp2 = matrixTensorOp2(currTree.dnodeW2{kidright-sl}, W2T2, hiddenSize);
        temp = temp1+temp2;
        currTree.dnodeW2{end}= currTree.dnodeW2{end} + bsxfun(@times, temp, dsig);
        
        temp1 = matrixMatrixOp(currTree.dnodebw1{kidleft-sl}, W1, hiddenSize);
        temp2 = matrixMatrixOp(currTree.dnodebw1{kidright-sl}, W2, hiddenSize);
        temp = temp1+temp2;
        currTree.dnodebw1{end}= currTree.dnodebw1{end} + bsxfun(@times, temp, reshape(dsig,1,hiddenSize));
    end
end
end

function [gradW] = getLeafDerT(tensor, x,xprime, hiddenSize)

diff = xprime - x;
diff = reshape(diff,[1,1,hiddenSize]);
diff = repmat(diff,[hiddenSize hiddenSize 1]);

T1 = diff.*tensor;
gradW = sum(T1,3);

end

function [mat]=matrixTensorOp2(tensor, T2, hiddenSize)

T3=reshape(tensor,[hiddenSize hiddenSize 1 hiddenSize]);
T4 = repmat(T3, [1 1 hiddenSize 1]);
T5 = sum(T2 .* T4,4);
mat = reshape(T5, [hiddenSize hiddenSize hiddenSize]);

end

function [T2]=getT2(matrix, hiddenSize)
T1=reshape(matrix,[1 1 hiddenSize hiddenSize]);
T2=repmat(T1,[hiddenSize hiddenSize 1 1]);
end

function [mat]= matrixMatrixOp(tensor, matrix, hiddenSize)

T1=reshape(matrix,[1 hiddenSize hiddenSize]);
T2=repmat(T1,[hiddenSize 1]);
T3=reshape(tensor,[hiddenSize 1 hiddenSize]);
T4 = repmat(T3, [1 hiddenSize 1]);
mat = sum(T2 .* T4,3);

end

function [gradW] = getLeafDerM(matrix, x,xprime, hiddenSize)

diff = xprime - x;
diff = reshape(diff,[1,hiddenSize]);
diff = repmat(diff,[hiddenSize 1]);

T1 = diff.*matrix;
gradW = sum(T1,2);

end

%         if(d1 > 0)
%             p1 = getVderW(W1,W2,bw1,p1,hiddenSize,W1T2,W2T2);
%             for i=1:1:hiddenSize
%                 for j=1:1:hiddenSize
%                     for k=1:1:hiddenSize
%                         gradW1(i,j) = gradW1(i,j) - t1.nodeFeaturesforward(i,end)*t2.dnodeW1{end}(i,j,k);
%                         gradW1(i,j) = gradW1(i,j) - t2.nodeFeaturesforward(i,end)*t1.dnodeW1{end}(i,j,k);
%                         gradW1(i,j) = gradW1(i,j) + t1.nodeFeaturesforward(i,end)*p1.dnodeW1{end}(i,j,k);
%                         gradW1(i,j) = gradW1(i,j) + p1.nodeFeaturesforward(i,end)*t1.dnodeW1{end}(i,j,k);
%                     end
%                 end
%             end
%         end
%         
%         if(d2 > 0)
%             p2 = getVderW(W1,W2,bw1,p2,hiddenSize,W1T2,W2T2);
%             for i=1:1:hiddenSize
%                 for j=1:1:hiddenSize
%                     for k=1:1:hiddenSize
%                         gradW1(i,j) = gradW1(i,j) - t1.nodeFeaturesforward(i,end)*t2.dnodeW1{end}(i,j,k);
%                         gradW1(i,j) = gradW1(i,j) - t2.nodeFeaturesforward(i,end)*t1.dnodeW1{end}(i,j,k);
%                         gradW1(i,j) = gradW1(i,j) + t2.nodeFeaturesforward(i,end)*p2.dnodeW1{end}(i,j,k);
%                         gradW1(i,j) = gradW1(i,j) + p2.nodeFeaturesforward(i,end)*t2.dnodeW1{end}(i,j,k);
%                     end
%                 end
%             end
%         end