function g = sigmoid(z)
% Finds the sigmoid of z

g = 1.0 ./ (1.0 + exp(-z));
end
