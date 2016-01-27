function [theta, cost] = AGWords_test(params, hiddenSize, words, We_orig)

We_old = We_orig;
pairs = getPairs(params.data,words,We_orig);

fprintf('Initial cost: %d\n',objectiveWordDerNew(hiddenSize, params.data, pairs, words, We_old, We_orig, params.lambda_t,params.lambda_w, params.margin,1));

Gw = zeros(size(We_orig));
delta = 1E-4;

for(i=1:1:params.epochs)
    j=1;
    params.data = params.data(randperm(numel(params.data)));
    while(j <= length(params.data))
        batch= {};
        e = params.batchsize+j-1;
        if(length(params.data) < e)
            e=length(params.data);
        end
        for l=j:1:e
            batch(end+1)=params.data(l);
        end
        
        pairs = getPairs(batch, words, We_orig);
        
        gradWords = computeGradNewObjWords(batch, pairs, hiddenSize, words, We_old, We_orig, params.lambda_w, params.margin);
        numgrad = computeNumericalGradient( @(x) objectiveWordDerNew(hiddenSize, batch, pairs, words, We_old, x, params.lambda_t,params.lambda_w, params.margin,0), We_orig(:));
        numgrad = reshape(numgrad,size(We_orig));
        diff = scoreGradient(numgrad, gradWords)
        Gw = Gw + gradWords.^2;
        
        j = j + params.batchsize;
        We_orig = We_orig - params.etaw*gradWords./(sqrt(Gw)+delta);
    end
    
    pairs = getPairs(params.data,words,We_orig);
    cost = objectiveWordDerNew(hiddenSize, params.data, pairs, words, We_old, We_orig, params.lambda_t,params.lambda_w, params.margin,1);
    fprintf('cost at epoch %i : %d\n',i,cost);
end

%cost = costf(theta);

end
