close all
clear all
clc


%% RK4
dt = 1/400;
N = 4001;
time = 0:dt:dt*(N-1);

% x0 = [1.95069; -0.08242; -5; -1]; 
x0 = [-0.5; -0.8 ; 0.3; 0.4];
% Initial conditions: [theta1, theta2, omega1, omega2]
[t, x] = ode113(@func, time, x0);

theta1 = x(:, 1)'; 
theta2 = x(:, 2)'; 

plot(time, theta1, 'linewidth', 3, 'DisplayName','\theta_1 true')
hold on;
plot(time, theta2, 'linewidth', 3, 'DisplayName','\theta_2 true')

save('data_val.mat', 'x');


%% PiSL0 Pred
[t, x] = ode113(@PiSL0, time, x0);

save('pred_PiSL0.mat', 'x')


%% PiSL2 Pred
[t, x] = ode113(@PiSL2, time, x0);

save('pred_PiSL2.mat', 'x')


%% PiSL5 Pred
[t, x] = ode113(@PiSL5, time, x0);

theta1 = x(:, 1)'; 
theta2 = x(:, 2)'; 

plot(time, theta1, 'linewidth', 3, 'DisplayName','\theta_1 pred')
hold on;
plot(time, theta2, 'linewidth', 3, 'DisplayName','\theta_2 pred')

save('pred_PiSL5.mat', 'x')

legend

hold off



function [dy] = PiSL0(t, y)

% The first order ODE: y = [theta1, theta2, omega1, omega2]
dy = zeros(4, 1);
dy(1) = y(3);
dy(2) = y(4);
dtheta = y(1) - y(2);

a = cos(dtheta);
b = y(4)^2*sin(dtheta);
c = sin(y(1));
d = y(3)^2*sin(dtheta);
e = sin(y(2));

dy(3) = 0.000687656*(6.78066*10e22*a*d - 7.2738*10e24*a*e + 5.23796*10e22*b + 3.30572*10e25*c) / (4.63705*10e19*a^2 - 2.1086*10e20);

dy(4) = -49.4936*(9.42361*10e17*a*b + 5.94731*10e20*a*c + 5.54728*10e18*d - 5.95071*10e20*e) / (4.63705*10e19*a^2 - 2.1086*10e20);

end


function [dy] = PiSL2(t, y)

% The first order ODE: y = [theta1, theta2, omega1, omega2]
dy = zeros(4, 1);
dy(1) = y(3);
dy(2) = y(4);
dtheta = y(1) - y(2);

a = cos(dtheta);
b = y(4)^2*sin(dtheta);
c = sin(y(1));
d = y(3)^2*sin(dtheta);
e = sin(y(2));

dy(3) = 0.6*(7.4268*10e8*a*d - 7.82813*10e10*a*e + 5.682*10e8*b + 3.59339*10e11*c) / (4.3538*10e8*a^2 - 2*10e9);

dy(4) = -2*(2.18112*10e8*a*b + 1.37938*10e11*a*c + 1.30961*10e9*d - 1.38038*10e11*e) / (4.3538*10e8*a^2 - 2*10e9);

end


function [dy] = PiSL5(t, y)

% The first order ODE: y = [theta1, theta2, omega1, omega2]
dy = zeros(4, 1);
dy(1) = y(3);
dy(2) = y(4);
dtheta = y(1) - y(2);

a = cos(dtheta);
b = y(4)^2*sin(dtheta);
c = sin(y(1));
d = y(3)^2*sin(dtheta);
e = sin(y(2));

dy(3) = (2.21372*10e9*a*d - 2.39322*10e11*a*e + 1.691*10e9*b + 1.07798*10e12*c) / (2.2302*10e9*a^2 - 1*10e10);

dy(4) = -2*(1.1077*10e9*a*b + 7.06134*10e11*a*c + 6.50215*10e9*d - 7.02937*10e11*e) / (2.2302*10e9*a^2 - 1*10e10);

end