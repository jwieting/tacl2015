function [output] = vectorsqrt(x)
    output = zeros(size(x));
    output(find(x<0)) = -sqrt(-x(find(x<0)));
    output(find(x>0)) = sqrt(x(find(x>0)));
end