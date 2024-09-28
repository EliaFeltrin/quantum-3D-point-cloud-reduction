import torch
import solver as sl
import cplex
# from cplex.exceptions import CplexError
import dimod
from dwave_qbsolv import QBSolv
from pyqubo import Array


def qubo_dict_from_matrix(Q):
    Q_dict = {}
    n = Q.shape[0]
    for i in range(n):
        for j in range(i, n):  # only upper triangular part (Q is symmetric)
            Q_dict[(i, j)] = Q[i, j].item()
    return Q_dict

# def al_cplex(Q_, A_, b_, mu_, lambda__, rho_, max_iter=1000):
#     m, n = A_.size()
#     Q = Q_.clone().float()

#     # Solving with Ax <= b, flip A and b for Ax >= b case
#     A = -A_.clone()
#     ATA = torch.matmul(torch.transpose(A, 0, 1), A)
#     b = torch.tensor([-b_ for _ in range(0, m)])
#     mu = mu_
#     lambda_ = lambda__.clone()
#     rho = rho_

#     iter = 0
#     c = torch.tensor([1])
#     al_min_p = None

#     while not (iter == max_iter or (c <= 0).sum().item() == m):
#         print(f'iter {iter}: ', end="")
#         Q += mu / 2 * ATA + torch.diag(torch.matmul((lambda_ - mu * b), A))

#         # Convert QUBO matrix into CPLEX-compatible format
#         Q_cplex = qubo_to_cplex_format(Q)

#         # Solve QUBO using CPLEX
#         try:
#             al_min_p = solve_qubo_cplex(Q_cplex, n)
#         except CplexError as exc:
#             print(f"Exception raised during CPLEX optimization: {exc}")
#             return torch.tensor([0 for _ in range(0, n)]).float(), mu, lambda_, iter
        
#         c = torch.matmul(A, al_min_p) - b

#         print(f'{(c <= 0).sum().item()}/{m} bounds satisfied. mu = {mu:.4f}, lambda = {lambda_.tolist()}')

#         lambda_ += mu * c * (c > 0).int()
#         mu *= rho
#         iter += 1

#     print("\n")
#     return al_min_p, mu, lambda_, iter


def qubo_to_cplex_format(Q):
    """ Converts a torch QUBO matrix to a format that CPLEX can use for QP. """
    Q = Q.detach().numpy().astype(float)
    num_vars = Q.shape[0]
    objective = []

    for i in range(num_vars):
        indices = []
        values = []
        for j in range(num_vars):
            if Q[i][j] != 0:
                indices.append(j)
                values.append(Q[i][j])
        # Append as two lists: [indices, values] for each variable
        objective.append([indices, values])

    return objective


def solve_qubo_cplex(Q_cplex, n):
    """ Uses CPLEX to solve a QUBO problem and returns the best solution. """

    # Create CPLEX problem
    prob = cplex.Cplex()
    # Suppress CPLEX output
    prob.set_log_stream(None)
    prob.set_error_stream(None)
    prob.set_warning_stream(None)
    prob.set_results_stream(None)
    # Define variables (binary decision variables)
    prob.variables.add(names=[f"x{i}" for i in range(n)], types=[prob.variables.type.binary] * n)
    # Set the quadratic objective function using the new format
    prob.objective.set_quadratic(Q_cplex)
    # Solve the QP problem using CPLEX
    prob.solve()
    # Retrieve the solution
    solution = prob.solution.get_values()
    # Convert solution to tensor
    solution_tensor = torch.tensor(solution).float()
    return solution_tensor

    

def al_dimod(Q_, A_, b_, mu_, lambda__, rho_, max_iter = 1000):
    m, n = A_.size()
    Q = Q_.clone()
    # solving with Ax <= b, it was c = Ax - b. With Ax >= b, we must flip A and b 
    A = -A_.clone()
    ATA = torch.matmul(torch.transpose(A, 0, 1), A)
    b = torch.tensor([-b_ for _ in range(0, m)])
    mu = mu_
    lambda_ = lambda__.clone()
    rho = rho_

    iter = 0
    c = torch.tensor([1])
    al_min_p = None

    sampler = dimod.SimulatedAnnealingSampler()

    while not (iter == max_iter  or (c <= 0).sum().item() == m):
        print(f'iter {iter}: ', end = "")
        Q += mu / 2 * ATA + torch.diag(torch.matmul((lambda_ - mu*b), A))

        qubo = qubo_dict_from_matrix(Q)

        #Solve using Simulated Annealing
        solution = sampler.sample_qubo(qubo, num_reads=100)
        

        #Get the best solution
        al_min_p = torch.tensor([solution.first.sample[i] for i in range(0, n)]).float()

        # al_min_p, val = sl.solve_bqp(Q, torch.tensor([]), torch.tensor([]))
        # al_min_p = torch.tensor(al_min_p, dtype = torch.float32)

        c = torch.matmul(A, al_min_p) - b

        print(f'{(c <= 0).sum().item()}/{m} bounds satisfied. mu = {mu:.4f}, lambda = {lambda_.tolist()}')

        lambda_ += mu * c * (c > 0).int()

        mu *= rho

        iter += 1


    print("\n")
    return al_min_p, mu, lambda_, iter

def al_qbsolv(Q_, A_, b_, mu_, lambda__, rho_, max_iter=1000):
    m, n = A_.size()
    Q = Q_.clone()
    
    # solving with Ax <= b, it was c = Ax - b. With Ax >= b, we must flip A and b 
    A = -A_.clone()
    ATA = torch.matmul(torch.transpose(A, 0, 1), A)
    b = torch.tensor([-b_ for _ in range(0, m)])
    mu = mu_
    lambda_ = lambda__.clone()
    rho = rho_

    iter = 0
    c = torch.tensor([1])
    al_min_p = None

    while not (iter == max_iter or (c <= 0).sum().item() == m):
        print(f'iter {iter}: ', end="")

        # Update Q matrix
        Q += mu / 2 * ATA + torch.diag(torch.matmul((lambda_ - mu * b), A))

        # Step 1: Build QUBO using PyQUBO
        # Define binary variables as an array of size `n`
        x = Array.create('x', shape=n, vartype='BINARY')

        # Define the QUBO matrix in PyQUBO's symbolic form
        qubo_expr = 0
        for i in range(n):
            for j in range(n):
                if Q[i, j] != 0:
                    qubo_expr += Q[i, j].item() * x[i] * x[j]

        # Compile the model
        model = qubo_expr.compile()

        # Convert PyQUBO model to QUBO matrix for solvers
        qubo, offset = model.to_qubo()

        # Step 2: Solve QUBO using QBSolv (Tabu Search)
        solution = QBSolv().sample_qubo(qubo)

        # Get the best solution and convert it to a tensor
        al_min_p = torch.tensor([solution.first.sample[f'x[{i}]'] for i in range(n)]).float()

        # Step 3: Compute constraints
        c = torch.matmul(A, al_min_p) - b

        print(f'{(c <= 0).sum().item()}/{m} bounds satisfied. mu = {mu:.4f}, lambda = {lambda_.tolist()}')

        # Step 4: Update lambda and mu
        lambda_ += mu * c * (c > 0).int()
        mu *= rho

        iter += 1

    print("\n")
    return al_min_p, mu, lambda_, iter

    
