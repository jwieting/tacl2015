function [pairs] = getPairsBatch(data,words,batchsize,We)
data = data(randperm(numel(data)));
pairs = {};
j=1;

while(j <= length(data))
    batch= {};
    e = batchsize+j-1;
    if(length(data) < e)
        e=length(data);
    end

    for l=j:1:e
        batch(end+1)=data(l);
    end

    j = j + batchsize;
    p = getPairs(batch, words, We);
    for i=1:1:length(p)
        pairs(end+1) = p(i);
    end
end

%for i=1:1:length(data)
%    pairs{i};
%    data{i};
%    [words(pairs{i}{1}.nums) words(data{i}{1}.nums)];
%end
end