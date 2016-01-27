function [grad] = computeGradNewObjWords(d, pairs, hiddenSize, words, We_old, We_orig, lambda, margin)

%grad = zeros(size(We_orig));
[num dim] = size(We_orig);
grad = sparse(num,dim);
for t=1:1:length(d)
    
    t1 = d{t}(1);
    t2 = d{t}(2);
    p1 = pairs{t}(1);
    p2 = pairs{t}(2);
    
    g1 = We_orig(:,t1);
    g2 = We_orig(:,t2);
    v1 = We_orig(:,p1);
    v2 = We_orig(:,p2);
    
    
    d1 = margin - sum(g1.*g2) + sum(v1.*g1);
    d2 = margin - sum(g1.*g2) + sum(v2.*g2);
    %gradWe = zeros(size(We_orig));
    gradWe = sparse(num,dim);
    
    if(d1 > 0 || d2 > 0)
        allwords = unique([t1 t2 p1 p2]);
        
        if(d1 > 0)
            for w=1:1:length(allwords)
                ww=allwords(w);
                
                m1 = zeros(hiddenSize);
                m2 = zeros(hiddenSize);
                m3 = zeros(hiddenSize);
                
                if(t1==ww)
                    m1 = eye(hiddenSize);
                end
                if(t2==ww)
                    m2 = eye(hiddenSize);
                end
                if(p1==ww)
                    m3 = eye(hiddenSize);
                end
                
                x = full(gradWe(:,ww));
                v = repmat(g2,[1 hiddenSize]);
                m = v.*m1;
                x = x - sum(m,1)';
                
                v = repmat(g1,[1 hiddenSize]);
                m = v.*m2;
                x = x - sum(m,1)';
                
                v = repmat(g1,[1 hiddenSize]);
                m = v.*m3;
                x = x + sum(m,1)';
                
                v = repmat(v1,[1 hiddenSize]);
                m = v.*m1;
                x = x + sum(m,1)';
                
                %                 for i=1:1:hiddenSize
                %                     for j=1:1:hiddenSize
                %                         gradWe(i,ww) = gradWe(i,ww) - m1(j,i)*g2(j);
                %                         gradWe(i,ww) = gradWe(i,ww) - m2(j,i)*g1(j);
                %                         gradWe(i,ww) = gradWe(i,ww) + m3(j,i)*g1(j);
                %                         gradWe(i,ww) = gradWe(i,ww) + m1(j,i)*v1(j);
                %                     end
                %                 end
                gradWe(:,ww)=x;
            end
        end
        
        if(d2 > 0)
            for w=1:1:length(allwords)
                ww=allwords(w);
                
                m1 = zeros(hiddenSize);
                m2 = zeros(hiddenSize);
                m3 = zeros(hiddenSize);
                
                if(t1==ww)
                    m1 = eye(hiddenSize);
                end
                if(t2==ww)
                    m2 = eye(hiddenSize);
                end
                if(p2==ww)
                    m3 = eye(hiddenSize);
                end
                x = full(gradWe(:,ww));
                v = repmat(g2,[1 hiddenSize]);
                m = v.*m1;
                x = x - sum(m,1)';
                
                v = repmat(g1,[1 hiddenSize]);
                m = v.*m2;
                x = x - sum(m,1)';
                
                v = repmat(g2,[1 hiddenSize]);
                m = v.*m3;
                x = x + sum(m,1)';
                
                v = repmat(v2,[1 hiddenSize]);
                m = v.*m2;
                x = x + sum(m,1)';
                %                 for i=1:1:hiddenSize
                %                     for j=1:1:hiddenSize
                %                         gradWe(i,ww) = gradWe(i,ww) - m1(j,i)*g2(j);
                %                         gradWe(i,ww) = gradWe(i,ww) - m2(j,i)*g1(j);
                %                         gradWe(i,ww) = gradWe(i,ww) + m3(j,i)*g2(j);
                %                         gradWe(i,ww) = gradWe(i,ww) + m2(j,i)*v2(j);
                %                     end
                %                 end
                gradWe(:,ww)=x;
            end
        end
    end
    
    grad = grad + gradWe;
end

grad = grad/length(d) - lambda.*(We_old-We_orig);

end