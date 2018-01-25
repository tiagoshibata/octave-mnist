function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

out = nnForwardProp(Theta1, Theta2, X);
[dummy, p] = max(out, [], 2);

end
