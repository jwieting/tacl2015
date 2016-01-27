function [ ] = AGWords_test(theta, params, hiddenSize, words, We_orig)

We_old = We_orig;

params.data = feedForwardTrees(params.data, theta, hiddenSize, We_orig);
[pairs, params.data] = getPairsBatch(params.data, words, params.batchsize);

fprintf('Initial cost: %d\n',objectiveWordDerNew(theta, hiddenSize, params.data, pairs, words, We_old, We_orig, params.lambda_t,params.lambda_w, params.margin,1));

Gt = zeros(length(theta),1);
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
        batch = feedForwardTrees(batch, theta, hiddenSize, We_orig);
        pairs = getPairs(batch, words);
        
        grad=computeGradNewObj(batch, pairs, theta, hiddenSize, words, We_orig, params.lambda_t,params.margin);
        
        numgrad = computeNumericalGradient( @(x) objectiveWordDerNew(x, hiddenSize, batch, pairs, words, We_old, We_orig, params.lambda_t,params.lambda_w, params.margin,0), theta);
        diff = scoreGradient(numgrad, grad)
        Gt = Gt + grad.^2;
        theta = theta - params.etat*grad./(sqrt(Gt)+delta);
        
        gradWords = computeGradNewObjWords(batch, pairs, theta, hiddenSize, words, We_old, We_orig, params.lambda_w, params.margin);
        numgrad = computeNumericalGradient( @(x) objectiveWordDerNew(theta, hiddenSize, batch, pairs, words, We_old, x, params.lambda_t,params.lambda_w, params.margin,0), We_orig(:));
        numgrad = reshape(numgrad,size(We_orig));
        diff = scoreGradient(numgrad, gradWords)
        Gw = Gw + gradWords.^2;
        
        j = j + params.batchsize;
        We_orig = We_orig - params.etaw*gradWords./(sqrt(Gw)+delta);
    end
    
    params.data = feedForwardTrees(params.data, theta, hiddenSize, We_orig);
    %pairs = getPairs(params.data, words);
    [pairs, params.data] = getPairsBatch(params.data, words, params.batchsize);
    cost = objectiveWordDerNew(theta, hiddenSize, params.data, pairs, words, We_old, We_orig, params.lambda_t,params.lambda_w, params.margin,1);
    fprintf('cost at epoch %i : %d\n',i,cost);
end

%cost = costf(theta);

end
