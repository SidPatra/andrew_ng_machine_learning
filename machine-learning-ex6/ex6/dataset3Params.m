function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

Cs = [0.01;0.03;0.1;0.3;1;3;10;30];
sigmas = [0.01;0.03;0.1;0.3;1;3;10;30];
pred_matrix = zeros(length(Cs),length(sigmas));
temp = [10000 0 0];
for i=1:size(Cs)(1)
  for j=1:size(sigmas)(1)
    model = svmTrain(X, y, Cs(i), @(x1, x2) gaussianKernel(x1, x2, sigmas(j))); %straight from ex6.m
    predictions = svmPredict(model,Xval); %given in line 19
    prediction_error = mean(double(predictions ~= yval)); %given in line 23
    pred_matrix(i,j) = prediction_error;
    %if temp(1) > prediction_error
    %  temp = [prediction_error i j];
    %endif
  endfor
endfor
[values, row_index]=min(pred_matrix);
[~ ,col] = min(values);
row = row_index(col);
 
C = Cs(row);
sigma = sigmas(col);
% =========================================================================

end
