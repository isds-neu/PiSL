% predicted function

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

function [dy] = pred(t, y)

% The first order ODE: y = [theta1, theta2, omega1, omega2]
dy = zeros(4, 1);
dy(1) = y(3);
dy(2) = y(4);
dtheta = y(1) - y(2);

num1 = 51.79054*y(3)^2*sin(2*dtheta) + 76.19862*y(4)^2*sin(dtheta) + 7133.49412*cos(y(2))*sin(dtheta) + 43849.30537*sin(y(1)) + 21.99995*y(3) - 7.6219*y(4)*cos(dtheta) - 162.55329 + 8.95456*sin(dtheta)^2;
den1 = -637 - 103.52781*sin(dtheta)^2;
dy(3) = num1/den1;

num2 = 4.27904*cos(y(2))*sin(dtheta) - 4.06734*sin(y(1)) + 39.84065*y(4)^2*sin(2*dtheta) + 774.44905*y(3)^2*sin(dtheta) + 53304.0283*cos(y(1))*sin(dtheta) + 26.81909*y(3)*cos(dtheta) - 61.33852*y(4) + 27.6182 + 4.55465*sin(dtheta)^2;
den2 = 490 + 79.66714*sin(dtheta)^2;
dy(4) = num2/den2;