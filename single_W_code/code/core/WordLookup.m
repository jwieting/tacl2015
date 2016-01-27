function index = WordLookup(InputString)
global wordMap
if wordMap.isKey(InputString)
    index = wordMap(InputString);
else
    if(wordMap.isKey('*UNKNOWN*'))
        index=wordMap('*UNKNOWN*');
    elseif(wordMap.isKey('UUUNKKK'))
        index=wordMap('UUUNKKK');
    end
end
