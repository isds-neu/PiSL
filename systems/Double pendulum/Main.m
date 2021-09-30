% slow-fast oscillator

close all
clear all
clc

%% RK4
dt = 1/400;
N = 4001;
time = 0:dt:dt*(N-1);

% x0 = [2; -0.1; -5; -1];        % Initial conditions: [theta1, theta2, omega1, omega2]
x0 = [1.95069; -0.08242; -5; -1];        % Initial conditions: [theta1, theta2, omega1, omega2]
%x0 = [1.95068789, -0.08242155, -21.156176, -19.89537];
% x0 = [-0.5; -0.8 ; 0.3; 0.4];
% x0 = [-0.2; -0.4 ; 0.1; 0.1];
[t, x] = ode113(@func, time, x0);
theta0 = x(:, 1:2)'; 
omega = x(:, 3:4); 

theta1 = x(:, 1)'; 
theta2 = x(:, 2)'; 

plot(time, theta1, 'linewidth', 3, 'DisplayName','\theta_1 true')
hold on;
plot(time, theta2, 'linewidth', 3, 'DisplayName','\theta_2 true')

%% shift theta
theta = theta0;
for i = 1:N
    if theta(1, i) > pi
        theta(1, i:end) = theta(1, i:end) - 2*pi;
    elseif theta(1, i) < -pi
        theta(1, i:end) = theta(1, i:end) + 2*pi;
    end
    if theta(2, i) > pi
        theta(2, i:end) = theta(2, i:end) - 2*pi;
    elseif theta(2, i) < -pi
        theta(2, i:end) = theta(2, i:end) + 2*pi;
    end
end
theta = theta(:, 1:2:end)';

save('data_ode113.mat', 'x')

% save('data_ode113S.mat', 'x')