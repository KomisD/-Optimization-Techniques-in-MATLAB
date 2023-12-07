
clc;
clear;
close all;



% Define the Objective Function
f = @(x,y) x.^3 .* exp(-x.^2 - y.^4);

% Define the Gradient Function
grad_f = @(x,y) [-3*x.^2 .* exp(-x.^2 - y.^4) + 2*x.^5 .* exp(-x.^2 - y.^4), ...
                 -4*y.^3 .* x.^3 .* exp(-x.^2 - y.^4)];

% Define the Hessian Function
second_grad_f = @(x, y) [ ...
    -6*x.*exp(-x.^2 - y.^4) + 6*x.^4.*exp(-x.^2 - y.^4) - 4*x.^7.*exp(-x.^2 - y.^4), ...
    -12*y.^3.*x.^2.*exp(-x.^2 - y.^4); ...
    -12*y.^3.*x.^2.*exp(-x.^2 - y.^4), ...
    -12*y.^2.*x.^3.*exp(-x.^2 - y.^4) + 16*y.^6.*x.^3.*exp(-x.^2 - y.^4) ...
];

epsilon = 1e-6; % More precise convergence criterion
gamma = 0.1; % Step size

starting_points = [0, 0; -1, -1; 1, 1];

fprintf('Minimization of f With Levenberg-Marquardt Method\n');
for i = 1:size(starting_points, 1)
    [xk, steps, points] = levenbergMarquardtMethod(epsilon, starting_points(i,:),gamma, f, grad_f, second_grad_f);
    fprintf('The lowest point found (%f, %f) with %f value and %d steps \n', xk(1), xk(2), f(xk(1), xk(2)), steps);
    fval = double(f(points(:,1), points(:,2)));
    k = 1:steps+1;
    figure(i);
    plot(k, fval);
    title(sprintf('Function value over iterations with Levenberg-Marquardt Method (Start: [%f, %f])', starting_points(i, 1), starting_points(i, 2)));
    xlabel('Iteration');
    ylabel('Function Value');
end


fprintf('----------------------------\n')
fprintf('Minimazation of f with dynamic gamma\n');
for i = 1:size(starting_points, 1)
    [xk, steps, points] = levenbergMarquardtMethodWithMinf(epsilon, starting_points(i,:), gamma, f, grad_f, second_grad_f);
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
    [xk, steps, points] = levenbergMarquardtMethodWithArmijo(epsilon, starting_points(i,:), f, grad_f, second_grad_f);
    fprintf('The lowest point found (%f, %f) with %f value and %d steps \n', xk(1), xk(2), f(xk(1), xk(2)), steps);
    fval = double(f(points(:,1), points(:,2)));
    k = 1:steps+1;
    figure(i +2*size(starting_points, 1));
    plot(k, fval);
    title(sprintf('Function value over iterations Using Armijo (Start: [%f, %f])', starting_points(i, 1), starting_points(i, 2)));
    xlabel('Iteration');
    ylabel('Function Value');
end

function [xk, steps, points] = levenbergMarquardtMethod(epsilon, startPoint,gamma , f, grad_f, second_grad_f)
    x = startPoint; % Initial guess
    steps = 0;
    points = x; % Initialize points as a row vector
    lambda = 1e-3; % Damping parameter
    lambda_up = 10; % Factor to increase lambda
    lambda_down = 10; % Factor to decrease lambda
    max_iter = 1000; % Maximum iterations to prevent infinite loops
    fprintf('Initial Point: (%f, %f)\n', x(1), x(2));
    
    while norm(grad_f(x(1), x(2))) > epsilon && steps < max_iter
        H = second_grad_f(x(1), x(2));
        g = grad_f(x(1), x(2));
        H_lm = H + lambda * eye(size(H)); % Modify Hessian for Levenberg-Marquardt
        step = H_lm \ g'; % Solve H_lm*delta = -g for delta
        
        % Handle potential numerical issues
        if rcond(H_lm) < 1e-15
            warning('Matrix is close to singular or badly scaled. Results may be inaccurate.');
            break;
        end
        
        x_new = x - gamma * step'; % Update step
        
        % Function value comparison for lambda adjustment
        if f(x_new(1), x_new(2)) < f(x(1), x(2))
            lambda = max(lambda / lambda_down, 1e-7); % Decrease lambda if successful
            x = x_new; % Update x
        else
            lambda = min(lambda * lambda_up, 1e7); % Increase lambda if not successful
        end

        points(end+1, :) = [x(1), x(2)];
        steps = steps + 1;
    end
    xk = x;
end


function [xk, steps, points] = levenbergMarquardtMethodWithMinf(epsilon, startPoint, gamma , f, grad_f, second_grad_f)
    x = startPoint; % Initial guess
    steps = 0;
    points = x; % Initialize points as a row vector
    lambda = 1e-3; % Damping parameter
    lambda_up = 10; % Factor to increase lambda
    lambda_down = 10; % Factor to decrease lambda
    max_iter = 1000; % Maximum iterations to prevent infinite loops
    fprintf('Initial Point: (%f, %f)\n', x(1), x(2));
    
    while norm(grad_f(x(1), x(2))) > epsilon && steps < max_iter
        H = second_grad_f(x(1), x(2));
        g = grad_f(x(1), x(2));
        H_lm = H + lambda * eye(size(H)); % Modify Hessian for Levenberg-Marquardt
        step = H_lm \ g'; % Solve H_lm*delta = -g for delta
        
        % Handle potential numerical issues
        if rcond(H_lm) < 1e-15
            warning('Matrix is close to singular or badly scaled. Results may be inaccurate.');
            break;
        end
        
        %calculate gamma
        d = -grad_f(x(1), x(2)); % Descent direction
        gamma = 1; % Reset gamma in each iteration
        x_new = x - gamma * step'; % Update step
        
        % Function value comparison for lambda adjustment
        if f(x_new(1), x_new(2)) < f(x(1), x(2))
            lambda = max(lambda / lambda_down, 1e-7); % Decrease lambda if successful
            x = x_new; % Update x
        else
            lambda = min(lambda * lambda_up, 1e7); % Increase lambda if not successful
        end

        points(end+1, :) = [x(1), x(2)];
        steps = steps + 1;
    end
    xk = x;
end

function [xk, steps, points] = levenbergMarquardtMethodWithArmijo(epsilon, startPoint, f, grad_f, second_grad_f)
    alpha = 0.01; % Armijo rule parameter
    beta = 0.5;   % Armijo rule reduction factor
    x = startPoint; % Initial guess
    steps = 0;
    points = x; % Initialize points as a row vector
    lambda = 1e-3; % Initial damping parameter
    lambda_up = 10; % Factor to increase lambda
    lambda_down = 10; % Factor to decrease lambda
    max_iter = 1000; % Maximum iterations to prevent infinite loops
    fprintf('Initial Point: (%f, %f)\n', x(1), x(2));
    
    while norm(grad_f(x(1), x(2))) > epsilon && steps < max_iter
        H = second_grad_f(x(1), x(2));
        g = grad_f(x(1), x(2));
        H_lm = H + lambda * eye(size(H)); % Modify Hessian for Levenberg-Marquardt
        delta = H_lm \ g'; % Solve H_lm*delta = -g for delta
        
        % Handle potential numerical issues
        if rcond(H_lm) < 1e-15
            warning('Matrix is close to singular or badly scaled. Results may be inaccurate.');
            break;
        end
        
        % Armijo Rule for step size selection
        gamma = 1; % Initial step size
        while f(x(1) + gamma * delta(1), x(2) + gamma * delta(2)) > f(x(1), x(2)) + alpha * gamma * g * delta
            gamma = beta * gamma; % Reduce step size
            if gamma < 1e-5 % Minimum threshold for gamma to avoid extremely small steps
                break;
            end
        end

        x_new = x - gamma * delta'; % Update step

        % Function value comparison for lambda adjustment
        if f(x_new(1), x_new(2)) < f(x(1), x(2))
            lambda = max(lambda / lambda_down, 1e-7); % Decrease lambda if successful
            x = x_new; % Update x
        else
            lambda = min(lambda * lambda_up, 1e7); % Increase lambda if not successful
        end

        points(end+1, :) = [x(1), x(2)];
        steps = steps + 1;
    end
    xk = x;
end
