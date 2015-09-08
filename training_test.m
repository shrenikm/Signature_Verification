% Code to test if the training data gives the proper output

% Loading
disp("Loading neural network parameters and training data...");
load("theta1.dat");
load("theta2.dat");
load("sig_data_X_2.dat");

disp(size(theta1));
disp(size(theta2));
disp(theta2(1));

X = sig_data_X_2;
m = size(X, 1);
X = [ones(m,1), X];

a1 = X;
z2 = a1*(theta1');
a2 = sigmoid(z2);
a2 = [ones(m,1),a2];
z3 = a2*(theta2');
a3 = sigmoid(z3);

disp("--------------------------------------------------------");
disp("Outputs to training data");
disp(a3);
