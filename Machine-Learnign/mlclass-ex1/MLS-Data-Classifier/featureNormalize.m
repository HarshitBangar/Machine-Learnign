function [X_norm, mu, sigma] = featureNormalize(X)
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));
[m,n] = size(X);
mu = 1/m*sum(X)';
sigma = std(X)';
X_norm = ((X - repmat(mu',m,1))./repmat(sigma',m,1));
end
