function numgrad = computeNumericalGradient(J, theta)

numgrad = zeros(size(theta));


EPSILON = 1E-4;

for i=1:1:length(numgrad)
    z = zeros(length(numgrad),1);
    z(i) = EPSILON;
    JplusEps = J(theta + z);
    JminusEps = J(theta - z);
    numgrad(i) = (JplusEps - JminusEps)/(2*EPSILON);
end

end
