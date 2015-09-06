function W  = debugInitialWeights(nout, nin)

% Initializes weights using a fixed strategy

W = zeros(nout, 1 + nin);

% Initialize W using "sin", this ensures that W is always of the same
% values and will be useful for debugging
W = reshape(sin(1:numel(W)), size(W)) / 10;

end;
