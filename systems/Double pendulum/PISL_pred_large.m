close all
clear all
clc


%% RK4
dt = 1/400;
N = 4001;
time = 0:dt:dt*(N-1);

x0 = [2; -0.05; -2; -0.8]; 

% Initial conditions: [theta1, theta2, omega1, omega2]
[t, x] = ode113(@func, time, x0);

theta1 = x(:, 1)'; 
theta2 = x(:, 2)'; 

plot(time, theta1, 'linewidth', 3, 'DisplayName','\theta_1 true')
hold on;
plot(time, theta2, 'linewidth', 3, 'DisplayName','\theta_2 true')

save('data_val_l.mat', 'x');


%% PiSL5 Pred
[t, x] = ode113(@PiSL5, time, x0);

theta1 = x(:, 1)'; 
theta2 = x(:, 2)'; 

plot(time, theta1, 'linewidth', 3, 'DisplayName','\theta_1 pred')
hold on;
plot(time, theta2, 'linewidth', 3, 'DisplayName','\theta_2 pred')

save('pred_PiSL5_l.mat', 'x')

legend

hold off


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