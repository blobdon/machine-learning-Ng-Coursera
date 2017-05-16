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

% feedforward network to get h2, which is h(x). h2 will be a m x k matrix
% h1 = sigmoid([ones(m, 1) X] * Theta1');
% h2 = sigmoid([ones(m, 1) h1] * Theta2');

% detailed feed-forward, required for backprop, which needs a1, a2, a3, z2
% 1. a1 = X plus ones = m x 401
a1 = [ones(m, 1) X];
% 2. z2 =  a1(m x 401) * T1'(401 x 25) = m x 25       
z2 = a1 * Theta1';      
% 3. a2 = g(z2) = m x 25  - item-wise sigmoid
a2 = sigmoid(z2);
% 4. add bias column of ones to a2, so a2 = m x 26
a2 = [ones(m, 1) a2];
% 5. z3 = a2(m x 26) * T2'(26 x 10) = m x 10
z3 = a2 * Theta2';
% 6. a3 = g(z3) = h(x) = m x 10  - item-wise sigmoid
a3 = sigmoid(z3);

% can we use a full m x k matrix of binary-based y-values?
y_binary = zeros(size(y,1), num_labels);
for k = 1 : num_labels
    y_binary(:,k) = (y == k);
endfor;

% loop through the examples (input) layer nodes), compute and total the costs
% we have to grab matching h(x) and y rows
% for i = 1:m
%     yi = y_binary(i,:); % 1 x k row of binary y-values
%     hxi = h2(i,:); % 1 x k row of predictions
%     temp = -yi * log(hxi)' - (1 - yi) * log(1 - hxi)';
%     J = J + temp;
% endfor;    
% J = (1/m) * J

% since the full y-matrix works, can i vectorize the cost function? yep, same result
% J = 0
% J_temp = -y_binary .* log(h2) - (1 - y_binary) .* log(1 - h2);
J_temp = -y_binary .* log(a3) - (1 - y_binary) .* log(1 - a3);
J = (1/m) * sum( J_temp(:) );
Jreg_temp = sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2));
J = J + ( lambda/(2 * m) * Jreg_temp );

% compute gradients
% d3 = (a3 (m x k) - y (m x k)) = m x k - k deltas for m examples
d3 = a3 .- y_binary;
% d2 = d3(m x k) * T2(k x hidden+1) = m x hidden+1 .* g`(z2(with ones)(m x hidden+1)) = m x hidden+1
d2 = d3 * Theta2 .* sigmoidGradient([ones(m,1) z2]);
% T2grad = d3' (k x m) * a2 (m x hidden+1) = k x hidden+1
Theta2_grad = (1/m) * d3' * a2;
% T1grad = d2'[1:](hidden x m) * a1 (m x input+1) = hidden x input+1
Theta1_grad = (1/m) * d2(:,2:end)' * a1;

% regularize gradients
Theta1_regterm = [zeros(size(Theta1,1),1) (lambda/m) * Theta1(:,2:end)];
Theta2_regterm = [zeros(size(Theta2,1),1) (lambda/m) * Theta2(:,2:end)];
Theta1_grad += Theta1_regterm;
Theta2_grad += Theta2_regterm;

% unroll all theta gradients into single vector
grad = [Theta1_grad(:); Theta2_grad(:)];














% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
