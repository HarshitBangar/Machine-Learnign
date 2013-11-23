function [jval, grad] = costFunction(theta)
% Initialize some useful values
data = load('mls_data.txt');
[nrows, ncols] = size(data);
X = data(:, 2:ncols); 
y = data(:, 1);

[X mu sigma] = featureNormalize(X); %Normalizing the feature for faster convergence

X = [ones(nrows, 1) X]; 
[m, n] = size(X); % number of training examples
jval = 0;
grad = 0;
jval = 1/(2*m)*sum((X*theta-y).^2);
grad = 1/m*sum((repmat((X*theta-y),1,n).*X));
