function [mat] = matrixLookup(map, key, hiddenSize)

    mat = [];
    if(isKey(map,key))
        mat = map(key);
    else
        mat = zeros(hiddenSize,hiddenSize);
    end
end