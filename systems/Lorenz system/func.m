% function

function [dy] = func(t, y)

alpha = 10;
beta = 8/3;
rho = 28;

% The first order ODE: y = [x, x_dot]
dy = zeros(3, 1);
dy(1) = alpha*(y(2) - y(1));
dy(2) = y(1)*(rho - y(3)) - y(2);
dy(3) = y(1)*y(2) - beta*y(3);