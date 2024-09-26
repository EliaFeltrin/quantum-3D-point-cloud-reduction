import torch
import random
import math

# Function to extract upper triangular elements (including diagonal)
def upper_triangular_indices(n):
    indices = []
    for i in range(n):
        for j in range(i, n):
            indices.append((i, j))
    return indices

def gen_Q(n, type, min_p_lst, min_val, res = 1000):


    if type == 'id':
        return torch.eye(n)
    elif type == 'diag':
        A = torch.tensor([t.tolist() for t in min_p_lst])
        A = torch.cat((A, torch.zeros(n-len(min_p_lst), n)), 0)
        b = torch.tensor([min_val for _ in range(0, len(min_p_lst))]).float()
        b = torch.cat((b, torch.tensor([0 for _ in range(0, n-len(min_p_lst))]))).float()


        Qd, _, _, _, = torch.linalg.lstsq(A, b)
        Q = torch.diag(Qd)

        for mp in min_p_lst:
            print(torch.matmul(mp, torch.matmul(Q, mp)))

        return Q
    
    elif type == 'triu':
        num_vectors = len(min_p_lst)
        
        # Total number of unknowns in the upper triangular part of Q (including diagonal)
        num_unknowns = (n * (n + 1)) // 2
        
        # Initialize matrix A and vector b
        A = torch.zeros((num_vectors, num_unknowns))
        b = torch.tensor([min_val for _ in range(0, num_vectors)], dtype=torch.float32)
        
        # Get the upper triangular indices of Q
        upper_indices = upper_triangular_indices(n)
        
        # Build the system of equations
        for idx, x in enumerate(min_p_lst):
            row = []
            for (i, j) in upper_indices:
                if i == j:
                    # Diagonal elements contribute x[i]^2
                    row.append(x[i])
                else:
                    # Off-diagonal elements contribute 2 * x[i] * x[j] since they appear twice in the quadratic form
                    row.append(x[i] * x[j])
            A[idx] = torch.tensor(row)
        
        # Solve the system A q = b to find the unknowns in the upper triangular part of Q
        # Use lstsq for a least-squares solution
        result = torch.linalg.lstsq(A, b)
        print(result)
        q, _, _, _ = result
        
        # Reconstruct the upper triangular matrix Q from the solution vector q
        Q = torch.zeros((n, n))
        for idx, (i, j) in enumerate(upper_indices):
            Q[i, j] = q[idx]
        
        print("Solved Q matrix:")
        print(Q)

        for mp in min_p_lst:
            print(torch.matmul(mp, torch.matmul(Q, mp)))
        


    