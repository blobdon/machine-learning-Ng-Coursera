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

% nn = 401 -> 26 -> 10, max(10) index is prediction
%   X is m x 400
% 1.add bias column of ones to X to get a1
a1 = [ones(m, 1) X];
%   a1 = m x 401
%   Theta1 is 25 x 401  
% 2. z2 =  a1(m x 401) * T1'(401 x 25) = m x 25       
z2 = a1 * Theta1';      
% 3. a2 = g(z2) = m x 25  - item-wise sigmoid
a2 = sigmoid(z2);
% 4. add bias column of ones to a2, so a2 = m x 26
a2 = [ones(m, 1) a2];
%   Theta2 is 10 x 26
% 5. z3 = a2(m x 26) * T2'(26 x 10) = m x 10
z3 = a2 * Theta2';
% 6. a3 = g(z3) = h(x) = m x 10  - item-wise sigmoid
a3 = sigmoid(z3);
% 7. max(a3) = index of max of each row = m x 1 = predicted label 
[~, p] = max(a3, [], 2);
% I want p is m x 1

% =========================================================================


end
