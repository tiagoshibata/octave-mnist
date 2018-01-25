indexes = randperm(size(X, 1), 100);
displayData(X(indexes, :));
display(reshape(predict(Theta1, Theta2, X(indexes, :)), 10, 10)');