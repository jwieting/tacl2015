function [score] = scoreGradient(v1, v2)
    score = norm(v1-v2)/norm(v1+v2);
end