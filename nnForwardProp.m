function h = nnForwardProp(theta1, theta2, inputs)

% hidden_output = sigmoid(theta1 * [1 ; inputs(:)]);
% h = sigmoid(theta2 * [1 ; hidden_output]);

m = size(inputs, 1);

hidden_output = sigmoid([ones(m, 1) inputs] * theta1');
h = sigmoid([ones(m, 1) hidden_output] * theta2');

end