function [We_new, newdata, wordsW] = limitWords(data, We, words)


wordsT =[];
WordsW = [];
%make set of words
for i=1:1:length(data)
    t1 = data{i}(1);
    t2 = data{i}(2);
    
    wordsT(end+1)=t1;
    wordsT(end+1)=t2;
end

wordsT = unique(wordsT);

%remap trees
for i=1:1:length(data)
    data{i}(1)=find(wordsT==data{i}(1));
    data{i}(2)=find(wordsT==data{i}(2));
end

wordsW = words(wordsT);
We_new = We(:,wordsT);
newdata = data;
end