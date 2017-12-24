function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% should not include theta_0 (theta(1))
J = (1/m) * sum(-y .* log(sigmoid(X * theta)) - (1 - y) .* log(1 - sigmoid(X * theta))) + (lambda / (2*m)) * (sum(theta .^ 2) - theta(1) ^ 2);

num = size(X, 2);
grad = zeros(num, 1);

for j = 1 : num
	sum = 0;
	for i = 1 : m
		% calculate Sigma sum
		% xi = example set i
		xi = X(i, :);
		hx = sigmoid(xi * theta);
		sum = sum + (hx - y(i)) * xi(j);
	end
	if j == 1
		% gradient for theta_0
		grad(j) = sum / m;
	else
		% gradients for remaining thetas
		grad(j) = sum / m + lambda * theta(j) / m;
	end
end




% =============================================================

end
