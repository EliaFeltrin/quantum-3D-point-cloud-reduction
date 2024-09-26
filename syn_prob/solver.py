import numpy as np
import gurobipy as gp
from gurobipy import GRB
import sys
import os

def solve_bqp(Q, A, b):
    """
    Solves a Binary Quadratic Program (BQP) using Gurobi, with a quadratic objective and linear constraints.

    Args:
        Q (torch.Tensor): The `n x n` matrix representing the quadratic coefficients in the objective function.
        A (torch.Tensor): The `m x n` matrix representing the linear constraints on the binary variables.
        b (list or int): The right-hand side vector (of size `m`) for the constraints, or a scalar if all constraints have the same right-hand side.

    Returns:
        x_opt (numpy.ndarray): The optimal solution vector (binary values) if the model is solved optimally.
        obj_val (float): The objective function value at the optimal solution.
        If no optimal solution is found, the function returns `None, None`.
    """

    sys.stdout = open(os.devnull, 'w')      #output suppresion

    # Convert torch tensors Q and A to numpy arrays
    Q_np = Q.detach().cpu().numpy()  # Ensure it's on CPU and then convert to numpy
    n = Q_np.shape[0]
    m = Q_np.shape[0]

    A_np = A.detach().cpu().numpy()
    
    b_np = np.array([b for _ in range(0, m)])

    # Number of variables (length of x)
    
    # Create a new model
    model = gp.Model("binary_quadratic_program")
    
    # Create binary variables (x_i in {0, 1})
    x = model.addVars(n, vtype=GRB.BINARY, name="x")
    
    # Set the objective function: min x^T Q x
    # Since x^T Q x is a quadratic expression, we use the 'QuadExpr' in Gurobi
    obj = gp.QuadExpr()
    for i in range(n):
        for j in range(i, n):  # Upper triangular part only
            obj += Q_np[i, j] * x[i] * x[j]
    
    model.setObjective(obj, GRB.MINIMIZE)
    
    # Add constraints: Ax >= b
    for i in range(A_np.shape[0]):  # Iterate over each constraint
        expr = gp.LinExpr()
        for j in range(n):
            expr += A_np[i, j] * x[j]
        model.addConstr(expr >= b_np[i], name=f"constraint_{i}")
    
    # Optimize the model
    model.optimize()

    
    # Extract the optimal solution
    if model.status == GRB.OPTIMAL:
        x_opt = np.array([x[i].x for i in range(n)])
        print(f"Optimal solution: {x_opt}")
        print(f"Objective value: {model.ObjVal}")
        sys.stdout = sys.__stdout__
        return x_opt, model.ObjVal
    else:
        print("No optimal solution found.")
        sys.stdout = sys.__stdout__
        return None, None