function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


% This code is for cost function
H = X*theta;
error = H-y;
errsquare = error.^2;
errSum = 0;
for i = 1:m
    errSum = errSum + errsquare(i);
end

LinearReg = (1 / (2 * m)) * errSum;

reg = 0;
sqrTheta = theta(2:end) .* theta(2:end);
for i = 1:length(sqrTheta)
    reg = reg + sqrTheta(i);
end

regulize = (lambda / (2 * m)) * reg;

J = LinearReg + regulize;




% This code for graient descent
thetaWithoutZero = [ [ 0 ]; theta([2:length(theta)])];
grad = (1 / m )* sum(error .* X) + (lambda / m ) * thetaWithoutZero';


% =========================================================================

grad = grad(:);

end
