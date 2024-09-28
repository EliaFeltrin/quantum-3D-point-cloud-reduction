import torch
import solver as sl
import cplex
from cplex.exceptions import CplexError

def qubo_dict_from_matrix(Q):
    Q_dict = {}
    n = Q.shape[0]
    for i in range(n):
        for j in range(i, n):  # only upper triangular part (Q is symmetric)
            Q_dict[(i, j)] = Q[i, j].item()
    return Q_dict

def al_cplex(Q_, A_, b_, mu_, lambda__, rho_, max_iter=1000):
    m, n = A_.size()
    Q = Q_.clone().float()

    # Solving with Ax <= b, flip A and b for Ax >= b case
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
        Q += mu / 2 * ATA + torch.diag(torch.matmul((lambda_ - mu * b), A))

        # Convert QUBO matrix into CPLEX-compatible format
        Q_cplex = qubo_to_cplex_format(Q)

        # Solve QUBO using CPLEX
        al_min_p = solve_qubo_cplex(Q_cplex, n)

        c = torch.matmul(A, al_min_p) - b

        print(f'{(c <= 0).sum().item()}/{m} bounds satisfied. mu = {mu:.4f}, lambda = {lambda_.tolist()}')

        lambda_ += mu * c * (c > 0).int()
        mu *= rho
        iter += 1

    print("\n")
    return al_min_p, mu, lambda_, iter


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
    try:

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

    except CplexError as exc:
        print(f"Exception raised during CPLEX optimization: {exc}")
        return None
    



    
