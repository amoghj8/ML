function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

%Insert ones to X
X = [ones(m,1) X];

%Compute z 
z_2 = (Theta1*X');

%Find sigmoid
X_2 = sigmoid(z_2);

%Take transpose
X_2 = X_2'; %size changed to 5000x25

%Add ones for next layer
X_2 = [ones(size(X_2,1),1) X_2]; %size 5000x26

%Compute z
z_3 = X_2*Theta2';

%Calculate sigmoid
p_sigmoid = sigmoid(z_3);

%Find max with index
[p_max,i_max] = max(p_sigmoid,[],2);
p = i_max;

% =========================================================================


end
