import numpy as np
from math import pi, sqrt
import time
from typing import Literal
from numba import njit
import matplotlib.pyplot as plt


DENSITY1 = 200
DENSITY2 = 50
DENSITY = 1.225  # air density
CD = 0.57

t0 = 0
x0 = 0
y0 = 0

x_target = 20
y_target = 0
x_wall = 10


def F2_primal(Vx: float, Vy: float, drag_term: float) -> float:
    '''Acceleration in x.'''
    Ax = -drag_term * sqrt(Vx**2 + Vy**2) * Vx
    return Ax

def F4_primal(Vx: float, Vy: float, drag_term: float) -> float:
    '''Acceleration in y.'''
    Ay = - 9.81 - drag_term * sqrt(Vx**2 + Vy**2) * Vy
    return Ay

@njit
def F_primal(state: np.ndarray, drag_term: float) -> np.ndarray:
    '''
    Calculate the F vector for the Runge Kutta.

    Parameters
    ---------
    - state: [x, Vx, y, Vy]
    - drag_term: 0.5*ρ*Cd*A/m

    Return
    ------
    - F_vector: [Vx, Ax, Vy, Ay]
    '''

    Vx = state[1]  # dx/dt=Vx=F1
    Vy = state[3]  # F3
    c =  sqrt(Vx**2 + Vy**2)
    Ax = -drag_term * c * Vx  # F2
    Ay = -9.81 - drag_term * c * Vy  # F4

    return np.array([Vx, Ax, Vy, Ay], dtype=np.float64)
    
@njit
def F_adjoint(state: np.ndarray, drag_term: float, Vx: float, Vy: float, Ax: float, Ay: float, c: float) -> np.ndarray:
    '''
    Parameters
    ----------
    - state: [Ψ1, dΨ1/dt, Ψ2, dΨ2/dt]
    - a: drag_term = 0.5*ρ*Cd*A/m
    - c: sqrt(Vx^2 + Vy^2)

    Return
    ------
    - F_vector: [dΨ1/dt, d^2Ψ1/dt^2, dΨ2/dt, d^2Ψ2/dt^2]
    '''

    a = drag_term
    psi1 = state[0]
    Vpsi1 = state[1] 
    psi2 = state[2]
    Vpsi2 = state[3]

    term1 = a*Vx*Vy/c
    term2 = a*(Vx**3*Ay + Vy**3*Ax)/c**3

    Apsi1 = Vpsi1*a*(2*Vx**2 + Vy**2)/c + Vpsi2*term1 + psi1*a*(2*Vx**3*Ax + 3*Vx*Ax*Vy**2 + Vy**3*Ay)/c**3 + psi2*term2  # d^2Ψ1/dt^2 = F2
    Apsi2 = Vpsi2*a*(2*Vy**2 + Vx**2)/c + Vpsi1*term1 + psi2*a*(2*Vy**3*Ay + 3*Vy*Ay*Vx**2 + Vx**3*Ax)/c**3 + psi1*term2  # d^2Ψ2/dt^2 = F4

    return np.array([Vpsi1, Apsi1, Vpsi2, Apsi2], dtype=np.float64)

def solve_rk4(old_state: np.ndarray, timestep: float, F: callable, drag_term: float, *args) -> np.ndarray:
    '''Solve a ODE system with Runge Kutta of 4th order.'''
    k1 = timestep * F(old_state, drag_term, *args)
    k2 = timestep * F(old_state + 0.5*k1, drag_term, *args)
    k3 = timestep * F(old_state + 0.5*k2, drag_term, *args)
    k4 = timestep * F(old_state + k3, drag_term, *args)
    new_state = old_state + (k1 + 2*k2 + 2*k3 + k4)/6

    return new_state

def solve_primal(t0: float, boundary_conditions: np.ndarray, y_target: float, timestep: float, drag_term: float):
    '''
    Solve the primal equations with 4-order Runge Kutta for ODE systems.

    Parameters
    ---------
    - t0: initial time
    - boundary_conditions: [x0, Vx0, y0, Vy0]
    - y_target
    - timestep: Δt for Runge Kutta
    - drag_term: term used in equations. Sometimes, is referred as a.

    Return
    ------
    - time_points: all descrete times
    - states: [x, dx/dt, y, dy/dt] for each timestep
    - accels: [d^2x/dt^2, d^2y/dt^2] for each timestep
    '''

    time_points = np.array([t0], dtype=np.float64)
    states = [boundary_conditions] 
    accels = [np.array(
        [F2_primal(boundary_conditions[1], boundary_conditions[3], drag_term),
        F4_primal(boundary_conditions[1], boundary_conditions[3], drag_term)], 
        dtype=np.float64
        )]

    i = 0
    while states[i][2] >= y_target:
        s_new = solve_rk4(states[i], timestep, F_primal, drag_term)
        a_new = np.array(
            [F2_primal(s_new[1], s_new[3], drag_term), 
             F4_primal(s_new[1], s_new[3], drag_term)], dtype=np.float64)  # [Ax, Ay]
        
        if s_new[2] < y_target:
            break
        
        states.append(s_new)
        accels.append(a_new)
        time_points = np.append(time_points, t0 + (i+1)*timestep)
        i += 1
    
    return time_points, np.vstack(states), np.vstack(accels)

def solve_FAE(t0: float, boundary_conditions: np.ndarray, primal_states: np.ndarray, primal_accels: np.ndarray, timestep: float, drag_term: float, c_values: np.ndarray) -> np.ndarray:
    '''
    Solve the FAE (Field Adjoint Equations) with 4-order Runge Kutta for ODE systems.

    Parameters
    ----------
    - t0: initial time
    - boundary_conditions: [Ψ1(t=T), dΨ1(t=T)/dt, Ψ2(t=T), dΨ2(t=T)/dt]
    - primal_states: array with [x, Vx, y, Vy] at each timestep
    - primal_accels: array with [Ax, Ay] at each timestep 
    - timestep: Δt used in Runge Kutta

    Return
    ------
    states: [Ψ1, dΨ1/dt, Ψ2, dΨ2/dt] for each timestep
    '''

    n_timepoints = primal_states.shape[0]
    states = np.zeros(shape=(n_timepoints, 4), dtype=np.float64)  # [Ψ1, dΨ1/dt, Ψ2, dΨ2/dt]
    states[n_timepoints-1] = boundary_conditions

    for i in range(n_timepoints-1, t0, -1):  # solve backward in time
        args = (primal_states[i, 1], 
                primal_states[i, 3], 
                primal_accels[i, 0], 
                primal_accels[i, 1], 
                c_values[i])
        new_stage = solve_rk4(states[i], -timestep, F_adjoint, drag_term, *args)
        states[i-1] = new_stage

    return states

def sensitivity_derivative_F3(psi1: np.ndarray, psi2: np.ndarray, Vx: np.ndarray, Vy: np.ndarray, E: float, SD3_TERM: float, c_values: np.ndarray) -> np.ndarray:
    '''Calculate the value of the function inside the integral of the sensitivity derivative dF/dE (n=3).'''

    expr_1 = (psi1*Vx + psi2*Vy) * c_values
    expr_2 = SD3_TERM * (DENSITY1 - 1.5*DENSITY2 - DENSITY1*E) / ((DENSITY2*0.5 + DENSITY1*E)**(1/3) * (E + 0.5)**2)
    F3 = expr_1 * expr_2

    return F3

def trapezium_integral(time_points: np.ndarray, primal_states: np.ndarray, fae_states: np.ndarray, E: float, SD3_TERM: float, c_values: np.ndarray, timestepRK: float) -> float:
    '''Calculate the integral of a function F using the trapezium method.'''

    n_steps = time_points.shape[0] - 1
    h = (n_steps*timestepRK - t0) / n_steps

    F_values = sensitivity_derivative_F3(
        fae_states[:, 0],
        fae_states[:, 2],
        primal_states[:, 1],
        primal_states[:, 3],
        E,
        SD3_TERM,
        c_values
    )

    I = F_values[0] + 2 * np.sum(F_values[1:-1]) + F_values[-1]
    I *= h / 2

    return I

def regression(primal_states: np.ndarray, timestepRK: float, quantity: Literal['time', 'x_position', 'Vx', 'Vy']) -> float:
    '''Calculate the shot's expected total time and final x position, i.e., the time when y=0, with polynomial regression. The polyonym is of 3rd order and uses a set of the last 4 points T(y) and x(y) of the shot.'''

    n_timepoints = primal_states.shape[0]
    
    if quantity == 'time':
        total_time_n = (n_timepoints-1)*timestepRK
        b_vector = np.array([total_time_n-3*timestepRK, total_time_n-2*timestepRK, total_time_n-timestepRK, total_time_n], dtype=np.float64)
    elif quantity == 'x_position':
        b_vector = np.array([primal_states[-4, 0], primal_states[-3, 0], primal_states[-2, 0], primal_states[-1, 0]], dtype=np.float64)
    elif quantity == 'Vx':
        b_vector = np.array([primal_states[-4, 1], primal_states[-3, 1], primal_states[-2, 1], primal_states[-1, 1]], dtype=np.float64)
    elif quantity == 'Vy':
        b_vector = np.array([primal_states[-4, 3], primal_states[-3, 3], primal_states[-2, 3], primal_states[-1, 3]], dtype=np.float64)

    y_n3 = primal_states[n_timepoints-4, 2]  # y_{N-3}
    y_n2 = primal_states[n_timepoints-3, 2]
    y_n1 = primal_states[n_timepoints-2, 2]
    y_n = primal_states[n_timepoints-1, 2]
    A_matrix = np.array([[1, y_n3, y_n3**2, y_n3**3],
                         [1, y_n2, y_n2**2, y_n2**3],
                         [1, y_n1, y_n1**2, y_n1**3],
                         [1, y_n, y_n**2, y_n**3]], dtype=np.float64)
    x_vector = np.linalg.solve(A_matrix, b_vector)

    expected_quantity = x_vector[0] + x_vector[1]*y0 + x_vector[2]*y0**2 + x_vector[3]*y0**3

    return expected_quantity

def optimization_workflow_adjoint_FD(design_vars: np.ndarray, steepest_descent_step: float, timestepRK: float, epsilonFD: float, adjoint_tol: float, ALM_tol: float = 1e-2, max_iterations: int = 50000) -> np.ndarray:
    '''
    Optimization workflow for testing the derivatives calculated by the Adjoint and Finite Differences methods and the Steepest Descent optimizer.

    Parameters
    ---------
    - design_vars: design variables
    - steepest_descent_step: steepest descent step
    - timestepRK: Δt for Rungke Kutta
    - adjoint_tol: convergence tolerance for Adjoint method
    - ALM_tol: convergence tolerance for the constraint
    - max_iterations: if not converged before, stop the optimization process at this iteration

    Return
    -----
    - design_variables_all: [Vx0, Vy0, E] at each optimization cycle
    - derivatives_all_adjoint: [∂F/∂Vx0, ∂F/∂Vy0, ∂F/∂E] at each optimization cycle
    - derivatives_all_FD: [∂F/∂Vx0, ∂F/∂Vy0, ∂F/∂E] at each optimization cycle
    - final_primal_state: state [x, Vx, y, Vy] when the optimization process converged
    - final_primal_state_all: state [x, Vx, y, Vy] at the final point at each optimization cycle 
    - total_time_all: list with the total times of the shot calculated at each optimization cycle 
    '''

    SD3_TERM = pi**(1/3) * DENSITY * CD * (3/(4*DENSITY1*DENSITY2))**(2/3) / 6 # constant term used for the calculation of the Sensitivity Derivative 3 

    total_time_old = 0
    total_time_new = 0
    psi1_T = 0  
    psi2_T = 0

    derivatives_all_adjoint = []
    derivatives_all_FD = []  
    design_variables_all = []
    final_primal_state_all = []
    total_time_all = []

    omega = 10  # ALM factor
    omega_max = 50
    omega_scale_factor = 1.03
    omega_start_iters = 20  # iteration at which ω starts increasing

    ALM_iter = 1
    opt_iter = 1

    while True:
        print(f'ALM iteration: {ALM_iter}')

        while True:
            if opt_iter == max_iterations:
                print(f'\nREACHED MAXIMUM ITERATIONS')
                return np.array(design_variables_all), np.array(derivatives_all_adjoint), np.array(derivatives_all_FD), primal_states, np.array(final_primal_state_all), total_time_all
            
            print(f'    Adjoint iteration: {opt_iter} , F={total_time_old}')

            design_variables_all.append(design_vars.copy())

            E = design_vars[2]
            m = 0.5 + E
            volume = E/DENSITY2 + 0.5/DENSITY1
            A = pi * ((3*volume) / (4*pi))**(2/3)
            drag_term = 0.5 * DENSITY * CD * A / m  # sometimes is referred as 'a' in the equations 

            # Solve primal problem
            boundary_conditions_primal = np.array([x0, design_vars[0], y0, design_vars[1]], dtype=np.float64)
            time_points, primal_states, primal_accels = solve_primal(t0, boundary_conditions_primal, y_target, timestepRK, drag_term)
            final_primal_state_all.append(primal_states[-1])
            total_time_new = regression(primal_states, timestepRK, 'time')
            total_time_all.append(total_time_new)
            final_x = regression(primal_states, timestepRK, 'x_position')
            final_Vx = regression(primal_states, timestepRK, 'Vx')
            final_Vy = regression(primal_states, timestepRK, 'Vy')
            # primal_states = np.append(primal_states, np.array([final_x, final_Vx, 0, final_Vy]))

            # Solve FD
            # -> Solve primal [Vx0+ε, Vy0, E]
            boundary_conditions_primal = np.array([x0, design_vars[0]+epsilonFD, y0, design_vars[1]], dtype=np.float64)
            time_points_1, primal_states_1, primal_accels_1 = solve_primal(t0, boundary_conditions_primal, y_target, timestepRK, drag_term)
            total_time_1 = regression(primal_states_1, timestepRK, 'time')
            final_x_1 = regression(primal_states_1, timestepRK, 'x_position')

            # -> Solve primal [Vx0, Vy0+ε, E]
            boundary_conditions_primal = np.array([x0, design_vars[0], y0, design_vars[1]+epsilonFD], dtype=np.float64)
            time_points_2, primal_states_2, primal_accels_2 = solve_primal(t0, boundary_conditions_primal, y_target, timestepRK, drag_term)
            total_time_2 = regression(primal_states_2, timestepRK, 'time')
            final_x_2 = regression(primal_states_2, timestepRK, 'x_position')

            # -> Solve primal [Vx0, Vy0, E+ε]
            E_3 = design_vars[2] + epsilonFD
            m_3 = 0.5 + E_3
            volume_3 = E_3/DENSITY2 + 0.5/DENSITY1
            A_3 = pi * ((3*volume_3) / (4*pi))**(2/3)
            drag_term_3 = 0.5 * DENSITY * CD * A_3 / m_3  
            boundary_conditions_primal = np.array([x0, design_vars[0], y0, design_vars[1]], dtype=np.float64)
            time_points_3, primal_states_3, primal_accels_3 = solve_primal(t0, boundary_conditions_primal, y_target, timestepRK, drag_term_3)
            total_time_3 = regression(primal_states_3, timestepRK, 'time')
            final_x_3 = regression(primal_states_3, timestepRK, 'x_position')

            # -> Calculate FD derivatives
            constraint_x_FD = final_x - x_target
            d1_FD = (total_time_1-total_time_new)/epsilonFD + omega*constraint_x_FD*(final_x_1 - final_x)/epsilonFD
            d2_FD = (total_time_2-total_time_new)/epsilonFD + omega*constraint_x_FD*(final_x_2 - final_x)/epsilonFD
            d3_FD = (total_time_3-total_time_new)/epsilonFD + omega*constraint_x_FD*(final_x_3 - final_x)/epsilonFD
            derivatives_FD = np.array([d1_FD, d2_FD, d3_FD], dtype=np.float64)
            derivatives_all_FD.append(derivatives_FD)

            # Solve FAE
            n_timepoints = primal_states.shape[0]
            Vpsi1_T = omega * constraint_x_FD
            psi3 = -(1 + final_Vx*Vpsi1_T) / final_Vy
            Vpsi2_T = psi3
            boundary_conditions_fae = np.array([psi1_T, Vpsi1_T, psi2_T, Vpsi2_T])  # [Ψ1(t=T), dΨ1(t=T)/dt, Ψ2(t=T), dΨ2(t=T)/dt]
            c_values = np.sqrt(primal_states[:, 1]**2 + primal_states[:, 3]**2)
            fae_states = solve_FAE(t0, boundary_conditions_fae, primal_states, primal_accels, timestepRK, drag_term, c_values)

            # Sensitivity derivatives
            sd1 = -fae_states[0, 0]
            sd2 = -fae_states[0, 2]
            sd3 = trapezium_integral(time_points, primal_states, fae_states, E, SD3_TERM, c_values, timestepRK)
            derivatives = np.array([sd1, sd2, sd3], dtype=np.float64)
            derivatives_all_adjoint.append(derivatives)

            # Update design variables
            design_vars = design_vars - steepest_descent_step*derivatives
            if design_vars[2] < 0:
                design_vars[2] = 0

            loss = abs(total_time_new - total_time_old)
            if loss <= adjoint_tol:
                print(f'    Adjoint Converged at iteration {opt_iter} and ω={omega}.')
                opt_iter += 1
                break

            total_time_old = total_time_new
            opt_iter += 1

        if abs(constraint_x_FD) <= ALM_tol and abs(primal_states[n_timepoints-1, 2]) <= ALM_tol:
            print(f'\nOptimization algorithm converged at ALM iteration {ALM_iter}.')
            print(f'F={total_time_new:.4f}, F_loss={loss}, Design Variables: {design_vars}, x*={primal_states[-1, 0]:.4f}, y*={primal_states[-1, 2]:.4f}')
            break
        
        if ALM_iter >= omega_start_iters and omega <= omega_max:
            omega = min(omega*omega_scale_factor, omega_max)

        ALM_iter += 1

    return np.array(design_variables_all), np.array(derivatives_all_adjoint), np.array(derivatives_all_FD), primal_states, np.array(final_primal_state_all), total_time_all

def optimization_workflow_adjoint(design_vars: np.ndarray, steepest_descent_step: float, timestepRK: float, adjoint_tol: float, ALM_tol: float = 1e-2, max_iterations: int = 50000) -> np.ndarray:
    '''
    Optimization workflow with the Adjoint method for calculating the derivatives and the Steepest Descent optimizer.

    Parameters
    ---------
    - design_vars: design variables
    - steepest_descent_step: steepest descent step
    - timestepRK: Δt for Rungke Kutta
    - adjoint_tol: convergence tolerance for Adjoint method
    - ALM_tol: convergence tolerance for the constraint
    - max_iterations: if not converged before, stop the optimization process at this iteration

    Return
    -----
    - design_variables_all: [Vx0, Vy0, E] at each optimization cycle
    - final_primal_state: state [x, Vx, y, Vy] when the optimization process converged
    - final_primal_state_all: state [x, Vx, y, Vy] at the final point at each optimization cycle 
    - total_time_all: list with the total times of the shot calculated at each optimization cycle 
    '''

    SD3_TERM = pi**(1/3) * DENSITY * CD * (3/(4*DENSITY1*DENSITY2))**(2/3) / 6 # constant term used for the calculation of the Sensitivity Derivative 3 

    total_time_old = 0
    total_time_new = 0
    psi1_T = 0  
    psi2_T = 0

    derivatives_all_adjoint = []
    design_variables_all = []
    final_primal_state_all = []
    total_time_all = []

    omega = 10  # used for x_target constraint
    omega_max = 50
    omega_scale_factor = 1.03
    omega_start_iters = 20  # iteration at which ω starts increasing
    omega2 = 1  # used for window constraint

    ALM_iter = 1
    opt_iter = 1

    while True:
        print(f'ALM iteration: {ALM_iter}')

        while True:
            if opt_iter == max_iterations:
                print(f'\nREACHED MAXIMUM ITERATIONS')
                return np.array(design_variables_all), primal_states, np.array(final_primal_state_all), total_time_all
            
            # print(f'    Adjoint iteration: {opt_iter}')

            design_variables_all.append(design_vars.copy())

            E = design_vars[2]
            m = 0.5 + E
            volume = E/DENSITY2 + 0.5/DENSITY1
            A = pi * ((3*volume) / (4*pi))**(2/3)
            drag_term = 0.5 * DENSITY * CD * A / m  # sometimes is referred as 'a' in the equations 

            # Solve primal problem
            boundary_conditions_primal = np.array([x0, design_vars[0], y0, design_vars[1]], dtype=np.float64)
            time_points, primal_states, primal_accels = solve_primal(t0, boundary_conditions_primal, y_target, timestepRK, drag_term)
            final_primal_state_all.append(primal_states[-1])
            total_time_new = regression(primal_states, timestepRK, 'time')
            total_time_all.append(total_time_new)

            # Solve FAE
            n_timepoints = primal_states.shape[0]
            target_constraint_x = primal_states[n_timepoints-1, 0] - x_target
            Vpsi1_T = omega * target_constraint_x
            psi3 = -(1 + primal_states[n_timepoints-1, 1]*Vpsi1_T) / primal_states[n_timepoints-1, 3]
            Vpsi2_T = psi3
            boundary_conditions_fae = np.array([psi1_T, Vpsi1_T, psi2_T, Vpsi2_T])  # [Ψ1(t=T), dΨ1(t=T)/dt, Ψ2(t=T), dΨ2(t=T)/dt]
            c_values = np.sqrt(primal_states[:, 1]**2 + primal_states[:, 3]**2)
            fae_states = solve_FAE(t0, boundary_conditions_fae, primal_states, primal_accels, timestepRK, drag_term, c_values)

            # Sensitivity derivatives
            sd1 = -fae_states[0, 0]
            sd2 = -fae_states[0, 2]
            sd3 = trapezium_integral(time_points, primal_states, fae_states, E, SD3_TERM, c_values, timestepRK)
            derivatives = np.array([sd1, sd2, sd3], dtype=np.float64)
            derivatives_all_adjoint.append(derivatives)

            # Handle window constraint
            x_wall_closest_index = abs(primal_states[:, 0] - x_wall).argmin()  # get the closest point to the wall
            x_wall_closest = primal_states[x_wall_closest_index, 0]  
            y_wall_closest = primal_states[x_wall_closest_index, 2]
            y_gap_below = max(0, 8 - y_wall_closest)
            y_gap_upper = max(0, y_wall_closest - 11)
            window_constraint_x = x_wall_closest - x_wall
            window_constraint = omega2 * (window_constraint_x**2 + y_gap_below**2 + y_gap_upper**2)

            if y_gap_below > 0:  # below window -> bigger Vy 
                sign = 1
            elif y_gap_upper > 0:  # upper window -> smaller Vy 
                sign = -1
            else: 
                sign = 0

            # Update design variables
            design_vars[0] = design_vars[0] - steepest_descent_step * derivatives[0]
            # design_vars[1] = design_vars[1] - steepest_descent_step * derivatives[1]
            design_vars[1] = design_vars[1] - steepest_descent_step * derivatives[1] + sign * steepest_descent_step * window_constraint
            design_vars[2] = design_vars[2] - steepest_descent_step * derivatives[2]

            omega2 = 10*opt_iter 

            if design_vars[2] < 0:
                design_vars[2] = 0

            loss = abs(total_time_new - total_time_old)
            if loss <= adjoint_tol:
                print(f'    Adjoint Converged at iteration {opt_iter} and ω={omega}.')
                opt_iter += 1
                break

            total_time_old = total_time_new
            if opt_iter % 100 == 0:
                print(f'    Adjoint iteration: {opt_iter}, F={total_time_new:.4f}, E={design_vars[2]:.4f}, x*={primal_states[-1, 0]:.4f}, y*={primal_states[-1, 2]:.4f}, x_wall={x_wall_closest:.4f}, y_wall={y_wall_closest:.4f}')
            # print(f'    Adjoint iteration: {opt_iter}, x*={primal_states[-1, 0]:.4f}, y*={primal_states[-1, 2]:.4f}, x_wall={x_wall_closest:.4f}, y_wall={y_wall_closest:.4f}')
            # print(f'    Adjoint iteration: {opt_iter}, F={total_time_new:.4f}, x*={primal_states[-1, 0]:.4f}, y*={primal_states[-1, 2]:.4f}')
            opt_iter += 1

        # if abs(target_constraint_x) <= ALM_tol and abs(primal_states[n_timepoints-1, 2]) <= ALM_tol:
        if abs(target_constraint_x) <= ALM_tol and abs(primal_states[n_timepoints-1, 2]) <= ALM_tol and y_gap_below <= ALM_tol and y_gap_upper <= ALM_tol and abs(window_constraint_x) <= ALM_tol:
        #     print(f'\nOptimization algorithm converged at ALM iteration {ALM_iter}.')
        #     print(f'F={total_time_new:.4f}, F_loss={loss}, Design Variables: {design_vars}, x*={primal_states[-1, 0]:.4f}, y*={primal_states[-1, 2]:.4f}')
            break
        
        if ALM_iter >= omega_start_iters and omega <= omega_max:
            omega = min(omega*omega_scale_factor, omega_max)

        ALM_iter += 1

    return np.array(design_variables_all), primal_states, np.array(final_primal_state_all), total_time_all

def independent_RK_step(timesteps: list, design_vars: np.ndarray):
    '''Conduct a timestep independence analysis for Runke Kutta.'''

    E = design_vars[2]
    m = 0.5 + E
    volume = E/DENSITY2 + 0.5/DENSITY1
    A = pi * ((3*volume) / (4*pi))**(2/3)
    drag_term = 0.5 * DENSITY * CD * A / m  # referred as a in the the equations 

    for dt in timesteps:
        boundary_conditions_primal = np.array([x0, design_vars[0], y0, design_vars[1]], dtype=np.float64)
        time_points, primal_states, primal_accels = solve_primal(t0, boundary_conditions_primal, y_target, dt, drag_term)
        print(f'Timestep {dt}: {primal_states[-1]}')

def independent_FD_step(steps: list, design_vars: np.ndarray, timestepRK: float):
    '''Conduct a step independence analysis for Finite Differences method.'''

    E = design_vars[2]
    m = 0.5 + E
    volume = E/DENSITY2 + 0.5/DENSITY1
    A = pi * ((3*volume) / (4*pi))**(2/3)
    drag_term = 0.5 * DENSITY * CD * A / m  # referred as a in the the equations
    omega = 20

    for epsilonFD in steps: 
        E = design_vars[2]
        m = 0.5 + E
        volume = E/DENSITY2 + 0.5/DENSITY1
        A = pi * ((3*volume) / (4*pi))**(2/3)
        drag_term = 0.5 * DENSITY * CD * A / m  # sometimes referred as a in the equations 

        # Solve primal problem
        boundary_conditions_primal = np.array([x0, design_vars[0], y0, design_vars[1]], dtype=np.float64)
        time_points, primal_states, primal_accels = solve_primal(t0, boundary_conditions_primal, y_target, timestepRK, drag_term)
        total_time = regression(primal_states, timestepRK, 'time')
        final_x = regression(primal_states, timestepRK, 'x_position')

        # Solve FD
        # -> Solve primal [Vx0+ε, Vy0, E] & [Vx0-ε, Vy0, E]
        boundary_conditions_primal = np.array([x0, design_vars[0]+epsilonFD, y0, design_vars[1]], dtype=np.float64)
        time_points_1, primal_states_1, primal_accels_1 = solve_primal(t0, boundary_conditions_primal, y_target, timestepRK, drag_term)
        total_time_1 = regression(primal_states_1, timestepRK, 'time')
        final_x_1 = regression(primal_states_1, timestepRK, 'x_position')
        boundary_conditions_primal = np.array([x0, design_vars[0]-epsilonFD, y0, design_vars[1]], dtype=np.float64)
        time_points_11, primal_states_11, primal_accels_11 = solve_primal(t0, boundary_conditions_primal, y_target, timestepRK, drag_term)
        total_time_11 = regression(primal_states_11, timestepRK, 'time')
        final_x_11 = regression(primal_states_11, timestepRK, 'x_position')

        # -> Solve primal [Vx0, Vy0+ε, E] & [Vx0, Vy0-ε, E]
        boundary_conditions_primal = np.array([x0, design_vars[0], y0, design_vars[1]+epsilonFD], dtype=np.float64)
        time_points_2, primal_states_2, primal_accels_2 = solve_primal(t0, boundary_conditions_primal, y_target, timestepRK, drag_term)
        total_time_2 = regression(primal_states_2, timestepRK, 'time')
        final_x_2 = regression(primal_states_2, timestepRK, 'x_position')
        boundary_conditions_primal = np.array([x0, design_vars[0], y0, design_vars[1]-epsilonFD], dtype=np.float64)
        time_points_22, primal_states_22, primal_accels_22 = solve_primal(t0, boundary_conditions_primal, y_target, timestepRK, drag_term)
        total_time_22 = regression(primal_states_22, timestepRK, 'time')
        final_x_22 = regression(primal_states_22, timestepRK, 'x_position')

        # -> Solve primal [Vx0, Vy0, E+ε] & [Vx0, Vy0, E-ε]
        E_3 = design_vars[2] + epsilonFD
        m_3 = 0.5 + E_3
        volume_3 = E_3/DENSITY2 + 0.5/DENSITY1
        A_3 = pi * ((3*volume_3) / (4*pi))**(2/3)
        drag_term_3 = 0.5 * DENSITY * CD * A_3 / m_3  
        boundary_conditions_primal = np.array([x0, design_vars[0], y0, design_vars[1]], dtype=np.float64)
        time_points_3, primal_states_3, primal_accels_3 = solve_primal(t0, boundary_conditions_primal, y_target, timestepRK, drag_term_3)
        total_time_3 = regression(primal_states_3, timestepRK, 'time')
        final_x_3 = regression(primal_states_3, timestepRK, 'x_position')

        E_33 = design_vars[2] - epsilonFD
        m_33 = 0.5 + E_33
        volume_33 = E_33/DENSITY2 + 0.5/DENSITY1
        A_33 = pi * ((3*volume_33) / (4*pi))**(2/3)
        drag_term_33 = 0.5 * DENSITY * CD * A_33 / m_33  
        boundary_conditions_primal = np.array([x0, design_vars[0], y0, design_vars[1]], dtype=np.float64)
        time_points_33, primal_states_33, primal_accels_33 = solve_primal(t0, boundary_conditions_primal, y_target, timestepRK, drag_term_33)
        total_time_33 = regression(primal_states_33, timestepRK, 'time')
        final_x_33 = regression(primal_states_33, timestepRK, 'x_position')

        # -> Calculate FD derivatives
        constraint_x = final_x - x_target
        # d1_FD = (total_time_1-total_time)/epsilonFD + omega*constraint_x*(final_x_1 - final_x)/epsilonFD
        # d2_FD = (total_time_2-total_time)/epsilonFD + omega*constraint_x*(final_x_2 - final_x)/epsilonFD
        # d3_FD = (total_time_3-total_time)/epsilonFD + omega*constraint_x*(final_x_3 - final_x)/epsilonFD

        d1_FD = (total_time_1-total_time_11)/(2*epsilonFD) + omega*constraint_x*(final_x_1 - final_x_11)/(2*epsilonFD)
        d2_FD = (total_time_2-total_time_22)/(2*epsilonFD) + omega*constraint_x*(final_x_2 - final_x_22)/(2*epsilonFD)
        d3_FD = (total_time_3-total_time_33)/(2*epsilonFD) + omega*constraint_x*(final_x_3 - final_x_33)/(2*epsilonFD)

        print(f'Epsilon {epsilonFD}: {d1_FD}, {d2_FD}, {d3_FD}')

def plot_derivatives_FD_adjoint(design_vars_all: np.ndarray, derivatives_all_adjoint: np.ndarray, derivatives_all_FD: np.ndarray):
    '''Create a plot with the derivatives calculated with the Adjoint and Finite Differences methods with their corresponding design variables values.'''

    x_labels = ['Vx0', 'Vy0', 'E']
    y_labels = ['∂F/∂Vx0', '∂F/∂Vy0', '∂F/∂E']

    plt.figure(figsize=(12, 5))
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.plot(design_vars_all[:, i], derivatives_all_FD[:, i], label="Finite Differences", marker="o", markersize=2, color='red')
        plt.plot(design_vars_all[:, i], derivatives_all_adjoint[:, i], label="Adjoint", marker="x", markersize=2, color='blue')
        plt.xlabel(f"{x_labels[i]}")
        plt.ylabel(f"{y_labels[i]}")
        plt.legend()
        plt.grid(True, alpha=0.6)

    plt.tight_layout()
    plt.show()

def plot_objective_convergence(total_time_all: list):
    '''Plot the objective function's (shot's expected duration) convergence through optimization cycles.'''
    plt.figure(figsize=(6, 3))
    plt.plot(np.arange(1, len(total_time_all)+1), total_time_all, color='red')
    plt.title('Objective Value Convergence')
    plt.xlabel('Optimization Cycle')
    plt.ylabel("Objective Value (Time) [s]")
    plt.grid(True, alpha=0.6)
    plt.show() 

def plot_orbit(final_primal_state: np.ndarray, plot_constraint: bool):
    '''Plot the orbit of the shot.'''
    plt.figure(figsize=(6, 3))

    if plot_constraint:
        plt.plot([10, 10], [0, 8], color='black')
        plt.plot([10, 10], [11, 15], color='black')

    plt.plot(final_primal_state[:, 0], final_primal_state[:, 2], color='red')
    plt.title("Shot's Orbit")
    plt.xlabel('x [m]')
    plt.ylabel("y [m]")
    plt.grid(True, alpha=0.6)
    plt.show() 

def plot_xy_convergence(final_primal_state_all: np.ndarray):
    '''Plot the final x and y positions' convergence through optimization cycles.'''

    iterations = final_primal_state_all.shape[0]
    iter_arr = np.arange(1, iterations+1)

    plt.figure(figsize=(6, 3))
    plt.plot(iter_arr, final_primal_state_all[:, 0], color='red')
    plt.title("Final X Position Convergence")
    plt.xlabel('Optimization Cycle')
    plt.ylabel("x [m]")
    plt.grid(True, alpha=0.6)
    plt.show() 

    plt.figure(figsize=(6, 3))
    plt.plot(iter_arr, final_primal_state_all[:, 2], color='blue')
    plt.title("Final Y Position Convergence")
    plt.xlabel('Optimization Cycle')
    plt.ylabel("y [m]")
    plt.grid(True, alpha=0.6)
    plt.show() 

def plot_designvars_convergence(design_variables_all: np.ndarray):
    '''Plot the Design Variables Convergence through optimization cycles.'''

    iterations = design_variables_all.shape[0]
    iter_arr = np.arange(1, iterations+1)

    plt.figure(figsize=(6, 3))
    plt.plot(iter_arr, design_variables_all[:, 0], color='red', label='Vx0')
    plt.plot(iter_arr, design_variables_all[:, 1], color='blue', label='Vy0')
    plt.title("Design Variables Vx0 and Vy0 Convergence")
    plt.xlabel('Optimization Cycle')
    plt.ylabel("V [m/s]")
    plt.grid(True, alpha=0.6)
    plt.legend()
    plt.show() 

    plt.figure(figsize=(6, 3))
    plt.plot(iter_arr, design_variables_all[:, 2], color='green')
    plt.title("Design Variable E Convergence")
    plt.xlabel('Optimization Cycle')
    plt.ylabel("E [kg]")
    plt.grid(True, alpha=0.6)
    plt.show() 

def main():
    ### RUNGE KUTTA'S TIMESTEP INDEPENDENCE ANALYSIS ###
    # init_design_vars = np.array([15, 5, 0.8], dtype=np.float64)  # [Vx0, Vy0, E]
    # timesteps = [0.01, 0.0075, 0.005, 0.0025, 0.001, 0.0009, 0.00085, 0.0008, 0.00075, 0.0005, 0.00025, 0.0001, 0.000075, 0.00005, 0.000025, 0.00001]
    # independent_RK_step(timesteps, init_design_vars)

    ### FINITE DIFFERENCES' STEP INDEPENDENCE ANALYSIS ###
    # init_design_vars = np.array([20, 7, 0.2], dtype=np.float64)  # [Vx0, Vy0, E]
    # steps = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13]
    # timestepRK = 0.0009
    # independent_FD_step(steps, init_design_vars, timestepRK)

    ### OPTIMIZATION WORKFLOW ADJOINT###
    init_design_vars = np.array([20, 20, 0.2], dtype=np.float64)  # [Vx0, Vy0, E]
    steepest_descent_step = 0.001
    timestepRK = 0.0009  # Δt for Runge Kutta
    epsilonFD = 0.001

    # design_variables_all, derivatives_all_adjoint, derivatives_all_FD, final_primal_state, final_primal_state_all, total_time_all = optimization_workflow_adjoint_FD(init_design_vars, steepest_descent_step, timestepRK, epsilonFD, adjoint_tol=1e-3)
    # np.savetxt("derivatives_all_adjoint.txt", derivatives_all_adjoint)
    # np.savetxt("derivatives_all_FD.txt", derivatives_all_FD)
    # plot_derivatives_FD_adjoint(design_variables_all, derivatives_all_adjoint, derivatives_all_FD)

    design_variables_all, final_primal_state, final_primal_state_all, total_time_all = optimization_workflow_adjoint(init_design_vars, steepest_descent_step, timestepRK, adjoint_tol=1e-6) 

    np.savetxt("design_variables.txt", design_variables_all)
    np.savetxt("final_primal_state.txt", final_primal_state)
    np.savetxt("final_primal_state_all.txt", final_primal_state_all)
    np.savetxt("total_time_all.txt", total_time_all)

    plot_objective_convergence(total_time_all)
    plot_orbit(final_primal_state, plot_constraint=True)
    # plot_orbit(final_primal_state, plot_constraint=False)
    plot_xy_convergence(final_primal_state_all)
    plot_designvars_convergence(design_variables_all)

if __name__=="__main__":
    main()