function [We_new, newdata, wordsW] = limitWords(data, We, words);
    

wordsT =[];
WordsW = [];
%make set of words
for i=1:1:length(data)
    t1 = data{i}{1};
    t2 = data{i}{2};
    
   arr = t1.nums;
   for j=1:1:length(arr)
       wordsT(end+1)=arr(j);
   end
   
   arr = t2.nums;
   for j=1:1:length(arr)
       wordsT(end+1)=arr(j);
   end
end

wordsT = unique(wordsT);

%remap trees
for i=1:1:length(data)
    t1 = data{i}{1};
    t2 = data{i}{2};
    
   arr = t1.nums;
   for j=1:1:length(arr)
       t1.nums(j)=find(wordsT==t1.nums(j));
   end
   
   arr = t2.nums;
   for j=1:1:length(arr)
       t2.nums(j)=find(wordsT==t2.nums(j));
   end
   data{i}{1} = t1;
   data{i}{2} = t2;
end

wordsW = words(wordsT);
We_new = We(:,wordsT);
newdata = data;