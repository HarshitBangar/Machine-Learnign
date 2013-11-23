%% Code for predicting Model
%
%  Author: Harshit Bangar
%  ------------
% 
%  This file contains code for checking predictability 
%  (Data is divided into training and testing)
%
%  Note: The code is an extension of my Coursera ML-Class work

%% Initialization
clear ; close all; clc

%%==============Reading the Data========%%
fprintf('Reading Input ... \n');
data = load('mls_data.txt');
[nrows, ncols] = size(data);
X = data(:, 2:ncols); 
y = data(:, 1);

[X mu sigma] = featureNormalize(X); %Normalizing the feature for faster convergence

X = [ones(nrows, 1) X]; %Adding the bias

%%===========Training Data=========%%
num_of_training_points = floor(0.7 * nrows);
Xtrain = X(1:num_of_training_points, :);
Ytrain = y(1:num_of_training_points, :);
%% Normal Equation
thetaNE = zeros(ncols, 1);
thetaNE = normalEqn(X, y);
fprintf('Displaying results with current data ... \n');
disp(1/(2*num_of_training_points)*sum((Xtrain*thetaNE-Ytrain).^2));

%%=============Testing Data=========%%
Xtrain = X(num_of_training_points+1:nrows, :);
Ytrain = y(num_of_training_points+1:nrows, :);
fprintf('Displaying results with future data ... \n');
disp(1/(2*(nrows-num_of_training_points))*sum((Xtrain*thetaNE-Ytrain).^2));
