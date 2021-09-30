% function

% function [dy] = func(y, epsilon)
% 
% % The first order ODE
% % dy = zeros(2, 1);
% % dy(1) = y(2) - y(1)^3;
% % dy(2) = -1/epsilon*(y(1) - y(2));
% 
% % The first order ODE
% dy = zeros(2, 1);
% dy(1) = y(2);
% dy(2) = 1/epsilon*(1 - y(1)^2)*y(2) - y(1);

function [dy] = func(t, y)

m1 = 35;   % g
m2 = 10;   % g
l1 = 9.1;  % cm
l2 = 7;    % cm
g = 981;   % cm/s^2

% The first order ODE: y = [theta1, theta2, omega1, omega2]
dy = zeros(4, 1);
dy(1) = y(3);
dy(2) = y(4);
dtheta = y(1) - y(2);
num1 = m2*l1*y(3)^2*sin(2*dtheta) + 2*m2*l2*y(4)^2*sin(dtheta) + 2*g*m2*cos(y(2))*sin(dtheta) + 2*g*m1*sin(y(1));
den1 = -2*l1*(m1 + m2*sin(dtheta)^2);
dy(3) = num1/den1;
num2 = m2*l2*y(4)^2*sin(2*dtheta) + 2*(m1 + m2)*l1*y(3)^2*sin(dtheta) + 2*g*(m1 + m2)*cos(y(1))*sin(dtheta);
den2 = 2*l2*(m1 + m2*sin(dtheta)^2);
dy(4) = num2/den2;