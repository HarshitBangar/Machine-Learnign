%% Code for predicting Model
%
%  Author: Harshit Bangar
%  ------------
% 
%  This file contains wrapper code for Predicting data problem
%  
%  Key Components: Identifying subset of features (featureDetect.m), Normalizing Features(featureNormalize.m), 
%  and Linear regression(ComputeCostMulti.m, gradientDescentMulti.m, and normalEqn.m).
%
%  The data is divided into training (50%), validation(30%) and testing(20%) . 
%
%  Note: The code is an extension of my Coursera ML-Class work

%% Initialization
clear ; close all; clc

%%==============================Reading the Data===================================%%
fprintf('Reading Input ... \n');
data = load('mls_data.txt');
[nrows, ncols] = size(data);
X = data(:, 2:ncols); 
y = data(:, 1);

[X mu sigma] = featureNormalize(X); %Normalizing the feature for faster convergence

X = [ones(nrows, 1) X]; %Adding the bias

fprintf('Program paused. Press enter to see Gradient descent.\n');
pause;

%%=============Breaking into cross-validation, training and testing sets==========%%
num_of_training_points = floor(0.5 * nrows);                   % The training set for developing models 
num_of_crossvalidation_points = floor(0.2*nrows);      % The cross validation set for selecting lambda

Xtrain = X(1:num_of_training_points, :);
ytrain = y(1:num_of_training_points, :);

Xcv = X(num_of_training_points+1:num_of_training_points+num_of_cv_points,:);
ycv = y(num_of_training_points+1:num_of_training_points+num_of_cv_points,:);

Xtest = X(num_of_cv_points+num_of_training_points+1:nrows,:);
ytest = y(num_of_cv_points+num_of_training_points+1:nrows,:);

%%======================Feature Subset Selection================================%%
%  We are interested in identifying features with high correlation with output and low inter class correlation
%  Sequential Feature Selection - Greedily growing from empty subset

prevcorrelationCoefficient = 0;
curcorrelationCoefficient = 0;
classCorrelationCoefficient = zeros(ncols, 1);
interCorrelationCoefficient = zeros(nrows, nrows);
finalTrainingSet = zeros(nrows, ncols);
Xtraindummy = Xtrain;
for i = 1:ncols
	classCorrelationCoefficient(i) = corrcoef(ytrain , Xtrain(: , i));
end

interCorrelationCoefficient = Xtrain' * Xtrain; 

% Populating the firstentry
firstentry = find(classCorrelationCoefficient == max(classCorrelationCoefficient));
finalTrainingSet(1,:) = Xtraindummy(:, firstentry);
prevcorrelationCoefficient = max(classCorrelationCoefficient);
Xtraindummy(:, firstentry) = Xtraindummy(:, ncols);
classCorrelationCoefficient(firstentry) = classCorrelationCoefficient(ncols);
interCorrelationCoefficient(firstentry, :) = interCorrelationCoefficient(ncols, :);
interCorrelationCoefficient(:, firstentry) = interCorrelationCoefficient(:, ncols); 
ncols = ncols - 1;
% Populating the other columns  - If the classCorrelationCoefficient decreases the method stops
for  i = 1:ncols
	
%%============================Linear Regression==================================%%

% Gradient descent
alpha = 0.1;
num_iters = 400;
thetaGD = zeros(ncols, 1);
[thetaGD, J_history] = gradientDescentMulti(X, y, thetaGD, alpha, num_iters);
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

fprintf('Program paused. Press enter to see Normal Equation.\n');
pause;

% Normal Equation
fprintf('Normal Equation Output ... \n');
thetaNE = zeros(ncols, 1);
thetaNE = normalEqn(X, y);
disp(1/(2*nrows)*sum((X*thetaNE-y).^2));

fprintf('Program paused. Press enter to see Conjugate Gradient.\n');
pause;

%%==========Conjugate Gradient(slow-need optimization in cost function 
%%                                  since it is making matrix calculation in each iteration)=======%%

options =optimset('GradObj', 'on', 'MaxIter', '100'); 
initialTheta = zeros(ncols, 1);
[optTheta,functionVal,exitFlag]=fminunc(@costFunction,initialTheta, options); 
fprintf('Conjugate Gradient Output ... \n');
disp(1/(2*nrows)*sum((X*optTheta-y).^2));
