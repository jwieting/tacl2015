function [ ] = AGWordsBigrams(theta, params, hiddenSize, words, We_orig, outfile)

We_old = We_orig;

params.data = feedForwardTrees(params.data, theta, hiddenSize, We_orig);
[pairs, params.data] = getPairsBatch(params.data, words, params.batchsize);

fprintf('Initial cost: %d\n',objectiveWordDerNew(theta, hiddenSize, params.data, pairs, words, We_old, We_orig, params.lambda_t,params.lambda_w, params.margin,1));

Gt = zeros(length(theta),1);
Gw = zeros(size(We_orig));
delta = 1E-4;

n=length(params.data);

ct=0;
q1=round(n/4);
q2=round(n/2);
q3=round(n*3/4);
for(i=1:1:params.epochs)
    j=1;
    params.data = params.data(randperm(numel(params.data)));
    while(j <= length(params.data))
        if(~params.quiet)
            fprintf('On example %d\n',j);
        end
        batch= {};
        e = params.batchsize+j-1;
        if(length(params.data) < e)
            e=length(params.data);
        end
        for l=j:1:e
            batch(end+1)=params.data(l);
        end
        if(length(batch)==1)
            j = j + params.batchsize;
            continue;
        end
        
        batch= feedForwardTrees(batch, theta, hiddenSize, We_orig);
        pairs = getPairs(batch, words);
        %[pairs, params.data] = getPairsBatch(params.data, words, params.batchsize);
        grad=computeGradNewObj(batch, pairs, theta, hiddenSize, words, We_orig,params.lambda_t,params.margin);
        Gt = Gt + grad.^2;
        theta = theta - params.etat*grad./(sqrt(Gt)+delta);
        gradWords = computeGradNewObjWords(batch, pairs, theta, hiddenSize, words, We_old, We_orig,params.lambda_w,params.margin);
        Gw = Gw + gradWords.^2;
        %numgrad = computeNumericalGradient( @(x) objectiveWordDer(x, hiddenSize, batch, words, We_orig), theta);
        %scoreGradient(numgrad,grad)
        %[numgrad(1:5) grad(1:5)];
        j = j + params.batchsize;
        We_orig = We_orig - params.etaw*gradWords./(sqrt(Gw)+delta);
        for k=j-params.batchsize:1:j
            if(k==q1)
                ct = ct+1;
                if(params.save)
                    save(strcat(strcat(strcat(outfile,'.params'),num2str(ct)),'.mat'),'theta','We_orig');
                end
            elseif(k==q2)
                ct = ct+1;
                if(params.save)
                    save(strcat(strcat(strcat(outfile,'.params'),num2str(ct)),'.mat'),'theta','We_orig');
                end
            elseif(k==q3)
                ct = ct+1;
                if(params.save)
                    save(strcat(strcat(strcat(outfile,'.params'),num2str(ct)),'.mat'),'theta','We_orig');
                end
            end
        end
    end
    
    params.data = feedForwardTrees(params.data, theta, hiddenSize, We_orig);
    [pairs, params.data] = getPairsBatch(params.data, words, params.batchsize);
    cost = objectiveWordDerNew(theta, hiddenSize, params.data, pairs, words, We_old, We_orig, params.lambda_t,params.lambda_w, params.margin,1);
    fprintf('cost at epoch %i : %d\n',i,cost);
    ct = ct + 1;
    if(params.save)
        save(strcat(strcat(strcat(outfile,'.params'),num2str(ct)),'.mat'),'theta','We_orig');
    end
end
