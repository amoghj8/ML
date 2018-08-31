function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
X = [ones(m,1) X];

y_matrix = eye(num_labels)(y,:) ;

z2 = X*Theta1';

a2 = sigmoid(z2);

a2 = [ones(size(a2,1),1) a2];

z3 = a2*Theta2';

a3 = sigmoid(z3);

for k=1:num_labels
    y_i = y==k;
    J = J + 1/m * sum(-1*(y_i).*log(a3(:,k))-((1-y_i).*log(1-(a3(:,k)))));    
end

sum_Theta1 = 0;
for k=2:size(Theta1,2)
  sum_Theta1 = sum_Theta1 + sum(Theta1(:,k).*Theta1(:,k));
end

sum_Theta2 = 0;
for k=2:size(Theta2,2)
  sum_Theta2 = sum_Theta2 + sum(Theta2(:,k).*Theta2(:,k));
end

J = J + (lambda/(2*m)) * (sum_Theta1+sum_Theta2);

% -------------------------------------------------------------

for t=1:m
  a1 = X(t,:); %1 x 401
  z2= a1*Theta1'; %1 x 25
  a2 = sigmoid(z2); %1 x 25
  z3 = [1,a2]*Theta2';% 1 x 10
  a3 = sigmoid(z3); % 1 x 10
  
  yi = ([1:num_labels]==y(t));
	d3 = a3 - yi;
  d2 = d3 * Theta2;
  d2 = d2(2:end);
  d2 = (d2).*sigmoidGradient(z2); %1 x 25
  
  Theta1_grad = Theta1_grad + d2' * a1; %  25x1    x    1x401
  Theta2_grad = Theta2_grad + d3' * [1,a2]; %  10x1    x    1x26  
end


Theta2_grad = Theta2_grad / m;
Theta1_grad = Theta1_grad / m;

Theta2_tmp = Theta2;
Theta1_tmp = Theta1;
Theta2_tmp(:, 1) = 0;
Theta1_tmp(:, 1) = 0;
Theta2_grad = Theta2_grad + lambda / m * Theta2_tmp;
Theta1_grad = Theta1_grad + lambda / m * Theta1_tmp;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
