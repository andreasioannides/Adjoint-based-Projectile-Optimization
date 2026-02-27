# Adjoint-based-Projectile-Optimization
This project addresses the solution of a constrained time-optimal projectile problem using the Continuous Adjoint Method. The objective is to determine the optimal launch conditions for a capsule containing pharmaceutical payload, released from ground level.

The capsule must pass through a prescribed window (box-shaped obstacle) and land at a specified target location in the shortest possible time. The design variables of the problem are:
- The initial horizontal velocity 𝑉𝑥
- The initial vertical velocity 𝑉𝑦​
- The mass of an additional internal inertial component

The optimization is performed using Stochastic Gradient Descent (SGD). The Augmented Lagrangian Method (ALM) is employed to enforce constraints.

Gradients of the objective function with respect to the design variables are computed using the Continuous Adjoint Method and validated against Finite Difference approximations to ensure accuracy.
