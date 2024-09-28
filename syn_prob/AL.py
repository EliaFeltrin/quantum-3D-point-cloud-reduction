import torch
import solver as sl
import dimod

def qubo_dict_from_matrix(Q):
    Q_dict = {}
    n = Q.shape[0]
    for i in range(n):
        for j in range(i, n):  # only upper triangular part (Q is symmetric)
            Q_dict[(i, j)] = Q[i, j].item()
    return Q_dict

def al(Q_, A_, b_, mu_, lambda__, rho_, max_iter = 1000):
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

    
