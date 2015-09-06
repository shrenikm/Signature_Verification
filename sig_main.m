% This is the main code that calculates the neural network values for signature verification

input_layer_size = 0;  % input features
hidden_layer_size = 125; % Number of nodes in the hidden layer
num_output = 1; %The number of classes is just one. It returns if the signature is the trained signature or not

disp("\n");
disp("---------------------------------------");
disp("Loading Data...");
disp("\n");
load("sig_data_X.dat");
load("sig_data_y.dat");

X = sig_data_X;
y = sig_data_y;

% size of the training data
m = size(sig_data_X, 1);
input_layer_size = size(sig_data_X, 2);


disp("Dimensions of training data:");
disp(m);
disp(input_layer_size);
disp("----------------------------------------");


disp("Generating random weights...");
disp("\n");
initialTheta1 = randInitialWeights(input_layer_size, hidden_layer_size);
initialTheta2 = randInitialWeights(hidden_layer_size, num_output);

% Unrolling the parameters
nn_params = [initialTheta1(:); initialTheta2(:)];

disp("Done generating weights");
disp("-----------------------------------------");

disp("\n");
disp("Feedforward...");
disp("\n");

lambda = 0;

J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_output, X, y, lambda);

disp("Cost without regularization: ");
disp(J);

disp("-----------------------------------------");
disp("\n");

disp("Cost after regularization: ");

lambda = 1;

[J, grad] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_output, X, y, lambda);
disp(J);

disp("------------------------------------------");
disp("\n");

disp("Checking backpropagation without regularization...");

checkNNGradients;

disp("\n");
disp("Enter to continue");
disp("-------------------------------------------");
pause;

disp("Checking backpropagation with regularization...");

lambda = 3;
checkNNGradients(lambda);

disp("Enter to continue");
disp("--------------------------------------------");
disp("\n");
pause;


% Now we train the neural network using fmincg

options = optimset('MaxIter', '50');

lambda = 1;

% short hand for the cost function

costFunction = @(p)nnCostFunction(p, input_layer_size, hidden_layer_size, num_output, X, y, lambda);

% Now the costFunction is a function that takes only one parameter as input, which is the neural network parameter

[final_nn_params, cost] = fmincg(costFunction, nn_params, options);

% Obtain theta1 and theta2 back from nn_params
theta1 = reshape(final_nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));

theta2 = reshape(final_nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_output, (hidden_layer_size + 1));

save theta1.dat theta1
save theta2.dat theta2

disp("\n");
disp("Press enter to continue");
disp("------------------------------------------------");
pause;

disp(size(theta1));
disp(size(theta2));
disp(cost);
































