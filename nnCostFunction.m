function [J grad] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_output, X, y, lambda)

% Computes the cost and the gradient of the neural network

% First we reshape nn_params to generate the weigths theta1 and theta2 of our 2 layer network

theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));

theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_output, (hidden_layer_size + 1));


m = size(X, 1);

J = 0;
theta1_grad = zeros(size(theta1));
theta2_grad = zeros(size(theta2));

% Adding the bias values to matrix X
X = [ones(m, 1), X];

% Finding cost without regularizing

for i=1:m

	% vectorized y
	yy = 1:num_output;
	yy = (yy == y(i));

	z2 = X(i, 1:end)*(theta1');
	a2 = sigmoid(z2);
	a2 = [1, a2];
	z3 = a2*(theta2');
	a3 = sigmoid(z3);

	ha = log(a3);
	hb = log(1-a3);
		
	for j=1:num_output
		
		J = J + ((-yy(j)*ha(j)) - ((1-yy(j))*(hb(j))));
	end;
	
end;

J = J/m;


% regularizing J

reg = 0;

theta1row = size(theta1, 1);
theta1col = size(theta1, 2);
theta2row = size(theta2, 1);
theta2col = size(theta2, 2);

for i=1:theta1row
	for j=2:theta1col
		
		% The first column of each parameter matrix consists of the values for the bias units. This is not included in the regularization
		reg = reg + theta1(i,j)^2;
	end;
end;

for i=1:theta2row
	for j=2:theta2col
		
		% Again, the terms corresponding to the bias values are not taken into account
		reg = reg + theta2(i,j)^2;
	end;
end;

reg = reg*(lambda/(2*m));

J = J + reg;



% Back Propagation to find the values of the gradient

a1 = X;
z2 = a1*(theta1');
a2 = sigmoid(z2);
a2 = [ones(m,1),a2];
z3 = a2*(theta2');
a3 = sigmoid(z3);

delta3 = a3 - eye(num_output)(y,:);
delta2 = (delta3*theta2(:, 2:end)).*sigmoidGradient(z2);

del1 = (delta2')*a1;
del2 = (delta3')*a2;

theta1_grad = (1/m)*del1;
theta2_grad = (1/m)*del2;


% Regularizing grad

theta1(:, 1) = zeros(theta1row, 1);
theta2(:, 1) = zeros(theta2row, 1);

theta1_grad = theta1_grad + (lambda/m)*(theta1);
theta2_grad = theta2_grad + (lambda/m)*(theta2);

%unroll gradients

grad = [theta1_grad(:) ; theta2_grad(:)];

end;




































