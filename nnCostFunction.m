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

% Reshape nn_params back into the parameters Theta1 and Theta2
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Auxiliary variable
m = size(X, 1);

J = 0;

y_mat = zeros(m, num_labels);
for i = 1:m
  y_mat(i, y(i)) = 1;
end

for i = 1:m
  expected = y_mat(i, :);
  output = nnForwardProp(Theta1, Theta2, X(i, :));
  J -= dot(expected, log(output)) + dot(ones(size(expected)) - expected, log(ones(size(output)) - output));
end
J += lambda / 2 * (sum(sum(Theta1(:, 2:end) .^ 2)) + sum(sum(Theta2(:, 2:end) .^ 2)));
J /= m;

% Gradient calculation
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
X = [ones(m, 1) X];
for i = 1:m
  in = X(i, :);
  z_2 = Theta1 * in';
  a_2 = [1 ; sigmoid(z_2)];

  z_3 = Theta2 * a_2;
  out = sigmoid(z_3);

  z_2 = [1 ; z_2];
  
  expected = y_mat(i, :);
  err_3 = out(:) - expected(:);
  err_2 = (Theta2' * err_3) .* sigmoidGradient(z_2);
  
  Theta2_grad += err_3 * a_2';
  Theta1_grad += err_2(2:end) * in;
end


% Unroll gradients
Theta1_grad(:, 2:end) += lambda * Theta1(:, 2:end);
Theta2_grad(:, 2:end) += lambda * Theta2(:, 2:end);
grad = [Theta1_grad(:) ; Theta2_grad(:)] ./ m;


end
