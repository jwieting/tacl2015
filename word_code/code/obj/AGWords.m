function [ ] = AGWords(params, hiddenSize, words, We_orig, outfile)

We_old = We_orig;
%pairs = getPairs(params.data,words,We_orig);
pairs = getPairsBatch(params.data,words,params.batchsize, We_orig);

fprintf('Initial cost: %d\n',objectiveWordDerNew(hiddenSize, params.data, pairs, words, We_old, We_orig, params.lambda_t,params.lambda_w, params.margin,1));

Gw = zeros(size(We_orig));
delta = 1E-4;

n=length(params.data);

ct=0;
q1=round(n/4);
q2=round(n/2);
q3=round(n*3/4);
for i=1:1:params.epochs
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
        
        pairs = getPairs(batch, words, We_orig);
        gradWords = computeGradNewObjWords(batch, pairs, hiddenSize, words, We_old, We_orig,params.lambda_w,params.margin);
        Gw = Gw + gradWords.^2;
        j = j + params.batchsize;
        We_orig = We_orig - params.etaw*gradWords./(sqrt(Gw)+delta);
        for k=j-params.batchsize:1:j
            if(k==q1)
                ct = ct+1;
                if(params.save)
                    save(strcat(strcat(strcat(outfile,'.params'),num2str(ct)),'.mat'),'We_orig');
                end
            end
            if(k==q2)
                ct = ct+1;
                if(params.save)
                    save(strcat(strcat(strcat(outfile,'.params'),num2str(ct)),'.mat'),'We_orig');
                end
            end
            if(k==q3)
                ct = ct+1;
                if(params.save)
                    save(strcat(strcat(strcat(outfile,'.params'),num2str(ct)),'.mat'),'We_orig');
                end
            end
        end
    end
    
    
    %pairs = getPairs(params.data,words,We_orig);
    pairs = getPairsBatch(params.data,words,params.batchsize, We_orig);
    cost = objectiveWordDerNew(hiddenSize, params.data, pairs, words, We_old, We_orig, params.lambda_t,params.lambda_w, params.margin,1);
    fprintf('cost at epoch %i : %d\n',i,cost);
    ct = ct + 1;
    if(params.save)
        save(strcat(strcat(strcat(outfile,'.params'),num2str(ct)),'.mat'),'We_orig');
    end
end
end
