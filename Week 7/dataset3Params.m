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

val_list = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

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

p_error = realmax;

for i= 1:size(val_list,2)
  for j=1:size(val_list,2)
    C_temp = val_list(1,i);
    sigma_temp = val_list(1,j);
    x1 = [1 2 1]; x2 = [0 4 -1]; 
    model= svmTrain(X, y, C_temp, @(x1, x2) gaussianKernel(x1, x2, sigma_temp));
    predictions = svmPredict(model,Xval);
    curr_error = mean(double(predictions ~= yval));
    if(curr_error < p_error)
      C = C_temp;
      sigma = sigma_temp;
      p_error = curr_error;
    end
  end
end





% =========================================================================

end
