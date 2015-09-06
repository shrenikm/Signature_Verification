function g = sigmoidGradient(z)

% Evaluates the gradient of the sigmoid function that is evaluated at z. Works if z is a vector or a matrix.
% This function is needed for back propagation

g = zeros(size(z));

g = sigmoid(z).*(1-sigmoid(z));

end
