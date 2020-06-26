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
disp(size(Theta1)) %4x3
disp(size(Theta2)) %4x5
disp(size(X)) %16x2
%disp(X)
%part0: make a1 (layer 1 by adding x0 column)
X=[ones(m,1) X] %16x3 (added column of ones)

%part1: find a2 (layer 2 calculations)
a2 = sigmoid(X*Theta1'); % 16x4
a2 = [ones(m,1) a2]; % 16x5 (added column of ones)
%part2: find a3 (layer 3 calculations)
a3 = sigmoid(a2*Theta2'); % 16x4
a=0;%dummy variable like in predictOneVsAll
[a,p] = max(a3,[],2);

%disp(a_1)





% =========================================================================


end
