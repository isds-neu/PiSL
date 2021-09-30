% Rossler oscillator

close all
clear all
clc


%% RK4
dt = 1/200;
N = 2001;
time = 0:dt:dt*(N-1);

X0 = [6 0 15];        % Initial condition

[t, X] = ode113(@func, time, X0);
save('data_val.mat', 'X')

x = X(:, 1); 
y = X(:, 2); 
z = X(:, 3); 

% trajectory
plot3(x, y, z, 'linewidth', 1.5)
hold on;

[t, X] = ode113(@f_PiSL, time, X0);
save('pred_PiSL.mat', 'X')

x = X(:, 1); 
y = X(:, 2); 
z = X(:, 3); 

% trajectory
plot3(x, y, z, 'linewidth', 1.5)
hold on;

[t, X] = ode113(@f_Eureqa, time, X0);
save('pred_Eureqa.mat', 'X')

x = X(:, 1); 
y = X(:, 2); 
z = X(:, 3); 

% trajectory
plot3(x, y, z, 'linewidth', 1.5)
hold on;

[t, X] = ode113(@f_SINDy, time, X0);
save('pred_SINDy.mat', 'X')

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


function [dy] = f_PiSL(t, y)

dy = zeros(3, 1);
dy(1) = -10.0638*y(1) + 10.0254*y(2);
dy(2) = 28.104*y(1) - 0.9933*y(2) - 1.0072*y(1)*y(3);
dy(3) = -2.6619*y(3) + 0.9943*y(1)*y(2);

end

function [dy] = f_SINDy(t, y)

dy = zeros(3, 1);
dy(1) = -0.456 - 9.177*y(1) + 9.168*y(2);
dy(2) = 22.323*y(1) + 0.146*y(2) - 0.846*y(1)*y(3);
dy(3) = 6.043 - 2.827*y(3) + 0.151*y(1)^2 + 0.813*y(1)*y(2);

end

function [dy] = f_Eureqa(t, y)

dy = zeros(3, 1);
dy(1) = -0.559 - 9.018*y(1) + 9.005*y(2);
dy(2) = -0.0471 + 18.788*y(1) + 1.855*y(2) - 0.0459*y(1)*y(2) - 0.741*y(1)*y(3);
dy(3) = -3.035 - 2.234*y(3) + 0.884*y(1)*y(2);

end