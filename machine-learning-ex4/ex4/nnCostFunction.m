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
%disp(size(X)) %16x2
%disp(size(y)) %16x1
%disp(size(Theta1)) %4x3
%disp(size(Theta2)) %4x5
a1 = [ones(m,1) X]; %all 1s in first col of a1, dimension is 16x3
z2 = a1*Theta1'; %16x3 x 3x4 = 16x4
%disp('test');
%disp(size(ones(size(z2),1))); %16x1
a2 = [ones(size(z2),1), sigmoid(z2)]; % dimension is 16x5;
z3 = a2*Theta2'; %16x5 x 5x4 = 16x4
a3 = sigmoid(z3); %16x4
hx = a3; %answer of neural network without regularization 16x4
%disp('hx');
%disp(size(hx));
vector_y = (1:num_labels)==y; %[[0100],[1000]...] for [2, 1,...] y results
J = sum( (vector_y .* log(hx)) + ((1-vector_y).*log(1-hx))); %inner sum from previous hw
J = -1/m * sum(J); %summing over itself

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
% Basically the same thing, but we need to do gradient checking and stuff from 
% the 
a_1 = [ones(m,1) X]; %all 1s in first col of a1, dimension is 16x3
z_2 = a_1*Theta1'; %16x3 x 3x4 = 16x4
%disp('test');
%disp(size(ones(size(z2),1))); %16x1
a_2 = [ones(size(z_2),1), sigmoid(z2)]; % dimension is 16x5;
z_3 = a_2*Theta2'; %16x5 x 5x4 = 16x4
a_3 = sigmoid(z_3); %16x4
h_x = a_3; %answer of neural network without regularization 16x4
vector_y = (1:num_labels)==y; %[[0100],[1000]...] for [2, 1,...] y results

%same as before until here. new part starts now:

d3 = a_3 - vector_y;
%disp(size(Theta2));
%disp(size(d3));
%disp(size(z2));
%disp(size(d3*Theta2));
%disp(size([ones(size(z_2,1),1) sigmoidGradient(z_2)]));
d2 = ((d3*Theta2) .* [ones(size(z_2,1),1) sigmoidGradient(z_2)]);
%disp(size(d2)); %16x5
d2 = d2(:,2:end);%getting rid of leftmost/bias column
disp(size(d2)); %16x4
disp(size(d3)); %16x4
disp(size(a_1));% 16x3
disp(size(a_2));% 16x5
%d2_ = d2'*[ones(size(a_1,1),1) a_1]; %5x4 x 4x16 <-wrong logic
%d3_ = d3'*a_2; <- wrong logic
Theta1_grad = (1/m) * (a_1'*d2)'; %( (16x3)' x 16x4 )' = (3x16 x 16x4)' = (3x4)' = 4x16 x 16x3 = 4x3
Theta2_grad = (1/m) * (a_2'*d3)'; %( (16x5)' x 16x4 )' = (5x16 x 16x4)' = (5x4)' = 4x16 x 16x5 = 4x5
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%grad_reg = lambda/m * theta(2:end); %from ex3
%Purpose of (:,2:end) is because we have to ignore bias terms on leftmost column.
J = J + ((lambda/(2*m)) * (sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2))));
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m)*Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m)*Theta2(:,2:end);
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
