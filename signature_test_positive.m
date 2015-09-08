% Code to test a single signature image that has already been preprocessed. (positive image)

%Loading the file
disp("Loading the image and neural network parameters...");
disp("------------------------------------");
disp("\n");

load("positive_example1.dat");
load("theta1.dat");
load("theta2.dat");

disp(theta2(1));

X = positive_example1;
disp(size(X));

m = size(X, 1);
X = [ones(m,1), X];

a1 = X;
z2 = a1*(theta1');
a2 = sigmoid(z2);
a2 = [ones(m,1),a2];
z3 = a2*(theta2');
a3 = sigmoid(z3);

disp("\n");
disp("----------------------------------");
disp("Output: ");
disp(a3);
