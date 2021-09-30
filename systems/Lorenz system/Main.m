% Rossler oscillator

close all
clear all
clc

% alpha = 10;
% beta = 8/3;
% rho = 28;

%% RK4
dt = 1/100;
N = 2001;
time = 0:dt:dt*(N-1);

X0 = [-8 7 27];        % Initial condition
[t, X] = ode113(@func, time, X0);
x = X(:, 1); 
y = X(:, 2); 
z = X(:, 3); 

% trajectory
plot3(x, y, z, 'linewidth', 1.5)
axis equal
xlabel('x')
ylabel('y')
zlabel('z')
grid on

save('Lorenz_data.mat', 'X')