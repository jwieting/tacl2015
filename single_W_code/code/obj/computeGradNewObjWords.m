function [grad] = computeGradNewObjWords(d, pairs, theta, hiddenSize, words, We_old, We_orig, lambda, margin)

W1 = reshape(theta(1:hiddenSize*hiddenSize),hiddenSize,hiddenSize);
W2 = reshape(theta(hiddenSize*hiddenSize+1:2*hiddenSize*hiddenSize),hiddenSize,hiddenSize);
bw1 = reshape(theta(2*hiddenSize*hiddenSize+1:2*hiddenSize*hiddenSize+hiddenSize),hiddenSize,1);

%lambda = 0;

%grad = zeros(size(We_orig));
[num dim] = size(We_orig);
grad = sparse(num,dim);
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
    %gradWe = zeros(size(We_orig));
    gradWe = sparse(num,dim);
    
    if(d1 > 0 || d2 > 0)
        allwords = unique([t1.nums t2.nums p1.nums p2.nums]);
        t1 = getVderW(W1,W2,bw1,t1,hiddenSize,We_orig);
        t2 = getVderW(W1,W2,bw1,t2,hiddenSize,We_orig);
        
        if(d1 > 0)
            p1 = getVderW(W1,W2,bw1,p1,hiddenSize,We_orig);
            for w=1:1:length(allwords)
                ww=allwords(w);
                m1 = zeros(hiddenSize);
                m2 = zeros(hiddenSize);
                m3 = zeros(hiddenSize);
                if(~isempty(t1.dnodeWe))
                    m1 = matrixLookup(t1.dnodeWe{end},ww,hiddenSize);
                elseif(t1.nums(1)==ww)
                    m1 = eye(hiddenSize);
                end
                if(~isempty(t2.dnodeWe))
                    m2 = matrixLookup(t2.dnodeWe{end},ww,hiddenSize);
                elseif(t2.nums(1)==ww)
                    m2 = eye(hiddenSize);
                end
                if(~isempty(p1.dnodeWe))
                    m3 = matrixLookup(p1.dnodeWe{end},ww,hiddenSize);
                elseif(p1.nums(1)==ww)
                    m3 = eye(hiddenSize);
                end

                x = full(gradWe(:,ww));
                v = repmat(t2.nodeFeaturesforward(:,end),[1 hiddenSize]);
                m = v.*m1;
                x = x - sum(m,1)';
                
                v = repmat(t1.nodeFeaturesforward(:,end),[1 hiddenSize]);
                m = v.*m2;
                x = x - sum(m,1)';
                
                v = repmat(t1.nodeFeaturesforward(:,end),[1 hiddenSize]);
                m = v.*m3;
                x = x + sum(m,1)';
                
                v = repmat(p1.nodeFeaturesforward(:,end),[1 hiddenSize]);
                m = v.*m1;
                x = x + sum(m,1)';
                
%                 for i=1:1:hiddenSize
%                     for j=1:1:hiddenSize
%                         x(i) = x(i) - m1(j,i)*t2.nodeFeaturesforward(j,end);
%                         x(i) = x(i) - m2(j,i)*t1.nodeFeaturesforward(j,end);
%                         x(i) = x(i) + m3(j,i)*t1.nodeFeaturesforward(j,end);
%                         x(i) = x(i) + m1(j,i)*p1.nodeFeaturesforward(j,end);
%                     end
%                 end
                gradWe(:,ww)=x;
            end
        end
        
        if(d2 > 0)
            p2 = getVderW(W1,W2,bw1,p2,hiddenSize,We_orig);
            for w=1:1:length(allwords)
                ww=allwords(w);
                m1 = zeros(hiddenSize);
                m2 = zeros(hiddenSize);
                m3 = zeros(hiddenSize);
                if(~isempty(t1.dnodeWe))
                    m1 = matrixLookup(t1.dnodeWe{end},ww,hiddenSize);
                elseif(t1.nums(1)==ww)
                    m1 = eye(hiddenSize);
                end
                if(~isempty(t2.dnodeWe))
                    m2 = matrixLookup(t2.dnodeWe{end},ww,hiddenSize);
                elseif(t2.nums(1)==ww)
                    m2 = eye(hiddenSize);
                end
                if(~isempty(p2.dnodeWe))
                    m3 = matrixLookup(p2.dnodeWe{end},ww,hiddenSize);
                elseif(p1.nums(1)==ww)
                    m3 = eye(hiddenSize);
                end
                
                x = full(gradWe(:,ww));
                v = repmat(t2.nodeFeaturesforward(:,end),[1 hiddenSize]);
                m = v.*m1;
                x = x - sum(m,1)';
                
                v = repmat(t1.nodeFeaturesforward(:,end),[1 hiddenSize]);
                m = v.*m2;
                x = x - sum(m,1)';
                
                v = repmat(t2.nodeFeaturesforward(:,end),[1 hiddenSize]);
                m = v.*m3;
                x = x + sum(m,1)';
                
                v = repmat(p2.nodeFeaturesforward(:,end),[1 hiddenSize]);
                m = v.*m2;
                x = x + sum(m,1)';
                %for i=1:1:hiddenSize
                %    for j=1:1:hiddenSize
                %        x(i) = x(i) - m1(j,i)*t2.nodeFeaturesforward(j,end);
                %        x(i) = x(i) - m2(j,i)*t1.nodeFeaturesforward(j,end);
                %        x(i) = x(i) + m3(j,i)*t2.nodeFeaturesforward(j,end);
                %        x(i) = x(i) + m2(j,i)*p2.nodeFeaturesforward(j,end);
                %    end
                %end
                gradWe(:,ww)=x;
            end
        end
    end

grad = grad + gradWe;
end

grad = grad/length(d) - lambda.*(We_old-We_orig);

end

function [currTree] = getVderW(W1,W2,bw1,currTree,hiddenSize, We_orig)

arr = unique(currTree.nums);

sl = size(currTree.nums,2);
%populate dsigmoids
for s=1:1:2*sl-1
    %take dsigmoid
    currTree.dsigmoid{end+1} = dsigmoid(currTree.nodeFeaturesforward(:,s));
end

for s=sl+1:1:2*sl-1
    m = containers.Map(arr(1),zeros(hiddenSize,hiddenSize));
    for i=2:1:length(arr)
       m(arr(i))=zeros(hiddenSize,hiddenSize); 
    end
    currTree.dnodeWe{end+1}= m;
end

%populate node forward tensors
for s=sl+1:1:2*sl-1
    kids = currTree.kids(s,:);
    kidleft = kids(1);
    kidright = kids(2);
    
    if(kidleft <= sl)
        w = currTree.nums(kidleft);
        currTree.dnodeWe{s-sl}(w)=W1 .* repmat(currTree.dsigmoid{s},[1 hiddenSize]);
    end
    
    if(kidright <= sl)
        w = currTree.nums(kidright);
        currTree.dnodeWe{s-sl}(w)=currTree.dnodeWe{s-sl}(w) + W2 .* repmat(currTree.dsigmoid{s},[1 hiddenSize]);
    end
    
     for k=1:1:length(arr)
         
         w=arr(k);
         %check what needs to be done.
         if(kidleft > sl && kidright <= sl)
             mleft = currTree.dnodeWe{kidleft-sl}(w);
             if(any(mleft(:)))
                 T1=reshape(mleft',[1 hiddenSize hiddenSize]);
                 T2=repmat(T1,[hiddenSize 1]);
                 T3=reshape(W1,[hiddenSize 1 hiddenSize]);
                 T4 = repmat(T3, [1 hiddenSize 1]);
                 mat = sum(T2 .* T4,3);
                 currTree.dnodeWe{s-sl}(w) = currTree.dnodeWe{s-sl}(w) + bsxfun(@times, mat, reshape(currTree.dsigmoid{s},1,hiddenSize)');
             end
             
         elseif(kidright > sl && kidleft <= sl)
             mleft = currTree.dnodeWe{kidright-sl}(w);
             if(any(mleft(:)))
                 T1=reshape(mleft',[1 hiddenSize hiddenSize]);
                 T2=repmat(T1,[hiddenSize 1]);
                 T3=reshape(W2,[hiddenSize 1 hiddenSize]);
                 T4 = repmat(T3, [1 hiddenSize 1]);
                 mat = sum(T2 .* T4,3);
                 currTree.dnodeWe{s-sl}(w) = currTree.dnodeWe{s-sl}(w) + bsxfun(@times, mat, reshape(currTree.dsigmoid{s},1,hiddenSize)');
             end
         elseif(kidright > sl && kidleft > sl)
             mat1=zeros(hiddenSize,hiddenSize);
             mat2=zeros(hiddenSize,hiddenSize);
             mleft = currTree.dnodeWe{kidleft-sl}(w);
             if(any(mleft(:)))
                 T1=reshape(mleft',[1 hiddenSize hiddenSize]);
                 T2=repmat(T1,[hiddenSize 1]);
                 T3=reshape(W1,[hiddenSize 1 hiddenSize]);
                 T4 = repmat(T3, [1 hiddenSize 1]);
                 mat1 = sum(T2 .* T4,3); 
             end
             mleft = currTree.dnodeWe{kidright-sl}(w);
             if(any(mleft(:)))
                 T1=reshape(mleft',[1 hiddenSize hiddenSize]);
                 T2=repmat(T1,[hiddenSize 1]);
                 T3=reshape(W2,[hiddenSize 1 hiddenSize]);
                 T4 = repmat(T3, [1 hiddenSize 1]);
                 mat2 = sum(T2 .* T4,3);
             end        
             currTree.dnodeWe{s-sl}(w) = currTree.dnodeWe{s-sl}(w) + bsxfun(@times, mat1+mat2, reshape(currTree.dsigmoid{s},1,hiddenSize)'); 
         end
     end
end
end