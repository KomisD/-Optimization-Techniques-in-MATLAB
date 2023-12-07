# Optimization-Techniques-in-MATLAB

## Overview
This repository contains MATLAB implementations of three optimization methods for unconstrained minimization of multivariable functions: Steepest Descent, Newton's Method, and the Levenberg-Marquardt Method. Developed by Dimos Kompitselidis, these algorithms showcase different approaches to minimize functions with varying parameters and initial conditions.

## Methods
### Steepest Descent
File: Steepest_Descent.m
Features: Utilizes gradient descent with adaptive step size. Ideal for functions with smooth gradients. Includes visualization of function values over iterations.
### Newton's Method
File: Newton_Method.m
Features: Employs second-order Taylor series expansion. Efficient for functions with predictable curvature. Demonstrates rapid convergence near minima.
### Levenberg-Marquardt Method
File: LevenbergMarquardt.m
Features: Combines gradient descent and Gauss-Newton method. Adaptable to functions with complex landscapes. Offers robustness against local minima.
Usage
To use these scripts, clone the repository and run the respective .m files in MATLAB. Ensure to adjust initial parameters and starting points as necessary for your specific function.

# Results and Analysis
Each method is analyzed for efficiency, convergence speed, and ability to avoid local minima. The documendation includes comprehensive results with graphical representations for each optimization technique under various scenarios.


