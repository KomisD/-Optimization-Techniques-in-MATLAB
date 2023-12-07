
clc;
clear;
close all;

% Define the Objective Function
f = @(x,y) x.^3 .* exp(-x.^2 -y.^4);

% Define the Gradient Function
grad_f = @(x,y) [-3*x.^2 .* exp(-x.^2 - y.^4) + 2*x.^5 .* exp(-x.^2 - y.^4), ...
                 -4*y.^3 .* x.^3 .* exp(-x.^2 - y.^4)];

epsilon = 0.01;
gamma = 0.1; % Step size

starting_points = [0, 0; -1, -1; 1, 1]; % Adjusted format for MATLAB

fprintf('Minimazation of f With Constant Gamma\n');
for i = 1:size(starting_points, 1)
    [xk, steps, points] = steepestDescent(epsilon, starting_points(i,:), f, gamma, grad_f);;
    fprintf('The lowest point found (%f, %f) with %f value and %d steps \n', xk(1), xk(2), f(xk(1), xk(2)), steps);
    fval = double(f(points(:,1), points(:,2)));
    k = 1:steps+1;
    figure(i);
    plot(k, fval);
    title(sprintf('Function value over iterations with constant Gamma (Start: [%f, %f])', starting_points(i, 1), starting_points(i, 2)));
    xlabel('Iteration');
    ylabel('Function Value');
end

fprintf('----------------------------\n')
fprintf('Minimazation of f with dynamic gamma\n');
for i = 1:size(starting_points, 1)
    [xk, steps, points] = steepestDescentWithMinf(epsilon, starting_points(i,:), f, grad_f);
    fprintf('The lowest point found (%f, %f) with %f value and %d steps \n', xk(1), xk(2), f(xk(1), xk(2)), steps);
    fval = double(f(points(:,1), points(:,2)));
    k = 1:steps+1;
    figure(i +size(starting_points, 1));
    plot(k, fval);
    title(sprintf('Function value over iterations with the dynamic Gamma (Start: [%f, %f])', starting_points(i, 1), starting_points(i, 2)));
    subtitle('Gamma that minimaze the asked function with an epsilon 0.001');
    xlabel('Iteration');

    ylabel('Function Value');
end

fprintf('----------------------------\n')
fprintf('Minimazation of f with Armijo\n');
for i = 1:size(starting_points, 1)
    [xk, steps, points] = steepestDescentWithArmigo(epsilon, starting_points(i,:), f, grad_f);
    fprintf('The lowest point found (%f, %f) with %f value and %d steps \n', xk(1), xk(2), f(xk(1), xk(2)), steps);
    fval = double(f(points(:,1), points(:,2)));
    k = 1:steps+1;
    figure(i +2*size(starting_points, 1));
    plot(k, fval);
    title(sprintf('Function value over iterations Using Armijo (Start: [%f, %f])', starting_points(i, 1), starting_points(i, 2)));
    xlabel('Iteration');
    ylabel('Function Value');
end



function [xk, steps, points] = steepestDescent(epsilon, startPoint, f, gamma, grad_f)
    x = startPoint; % Initial guess
    steps = 0;
    points = x; % Initialize points as a row vector
    fprintf('Initial Point: (%f, %f)\n', x(1), x(2));
    while norm(grad_f(x(1), x(2))) > epsilon
        x = x - gamma * grad_f(x(1), x(2)); % Update step
        points(end+1, :) = [x(1), x(2)];
        steps = steps + 1;
    end
    xk = x;
end

function [xk, steps, points] = steepestDescentWithMinf(epsilon, startPoint, f, grad_f)
    alpha = 0.01; % Line search parameter
    beta = 0.5;   % Line search parameter

    x = startPoint; % Initial guess
    steps = 0;
    points = x; % Initialize points as a row vector
    fprintf('Initial Point: (%f, %f)\n', x(1), x(2));
    while norm(grad_f(x(1), x(2))) > epsilon

        %calculate gamma
        d = -grad_f(x(1), x(2)); % Descent direction
        gamma = 1; % Reset gamma in each iteration
        while f(x(1) + gamma * d(1), x(2) + gamma * d(2)) > 0.001 %a random thresshold
            gamma = beta * gamma; % Reduce gamma
            if gamma < 0.01% Avoid infinite loop
                break;
            end
        end

     
        x = x + gamma * d; % Update step
        points(end+1, :) = [x(1), x(2)];
        steps = steps + 1;
    end
    xk = x;
end


function [xk, steps, points] = steepestDescentWithArmigo(epsilon, startPoint, f, grad_f)
    alpha = 0.01; % Line search parameter
    beta = 0.5;   % Line search parameter

    x = startPoint; % Initial guess
    steps = 0;
    points = x; % Initialize points as a row vector
    fprintf('Initial Point: (%f, %f)\n', x(1), x(2));
    while norm(grad_f(x(1), x(2))) > epsilon

        %calculate gamma
        d = -grad_f(x(1), x(2)); % Descent direction
        gamma = 1; % Reset gamma in each iteration
        %armigo rule
        while f(x(1) + gamma * d(1), x(2) + gamma * d(2)) > f(x(1), x(2)) + alpha * gamma * grad_f(x(1), x(2))' * d
            gamma = beta * gamma; % Reduce gamma
            if gamma < 0.01% Avoid infinite loop
                break;
            end
        end

     
        x = x + gamma * d; % Update step
        points(end+1, :) = [x(1), x(2)];
        steps = steps + 1;
    end
    xk = x;
end









