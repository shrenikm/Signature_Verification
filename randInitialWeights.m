function W = randInitialWeights(nin, nout)

% randomly initialize weights for a layer with 'nin' inputs and 'nout' outputs
% the first column values are the weights for the bias units

W = zeros(nout, nin+1);
epsilon_init = 0.12;
W = rand(nout, 1+nin)*2*epsilon_init - epsilon_init;

end;
