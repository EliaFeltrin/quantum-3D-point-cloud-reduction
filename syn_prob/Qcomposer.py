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

def gen_Q(n, type, min_p_lst, min_val, res = 1000, check = True, float_prec = 1e-4, coeff = 1):


    if type == 'id':
        return torch.eye(n), min_p_lst
    
    elif type == 'diag':
        if len(min_p_lst) == 1:
            Q = torch.zeros(n, n)
            
            missing_cost = min_val
            n_selected_point_at_m = min_p_lst[0].sum().item()
            min_p_one_indexes = [i for i,b in enumerate(min_p_lst[0]) if b == 1]
            for d in range(0,n):
                if d in min_p_one_indexes:
                    val = random.random() * missing_cost
                    Q[d,d] = val
                    missing_cost -= val 
                else:
                    Q[d,d] = (random.random() + 1 + float_prec*10) * coeff * min_val# / n_selected_point_at_m

            Q[min_p_one_indexes[0], min_p_one_indexes[0]] += missing_cost

            return Q, min_p_lst
        else:
            print("SORRY, not working...")
            exit()
            print("WARNING: YOU'RE TRYING TO GENERATE Q DIAGONAL WITH MORE THAN ONE GLOBAL MINIMUM:\nthis function may need more than one try and may reduce the number of gloabal minimum point. It also need to check all the candidate minimum point")
            A = torch.tensor([t.tolist() for t in min_p_lst])
            b = torch.tensor([min_val for _ in range(0, len(min_p_lst))]).float()
            b = torch.cat((b, torch.tensor([random.random()*min_val for _ in range(0, n-len(min_p_lst))]))).float()

            ok = False
            while not ok:
                ok = False
                A = torch.cat((A, torch.rand(n-len(min_p_lst), n)), 0)

                try:
                    Qd= torch.linalg.solve(A, b)
                    ok = True
                except RuntimeError:
                    print("The random part of the system needed to generate Q lead to a singular matrix. Let's try again...")

            Q = torch.diag(Qd)

            ok_ = True
            vals = []
            ok_ps = []
            for mp in min_p_lst:
                r = torch.matmul(mp, torch.matmul(Q, mp)).item()
                vals.append(r)
                if abs(r  - min_val) > float_prec:
                    ok_ = False
            if not ok_:
                print("The eq system needed to determine Q was solved wrongly. An updated list of global min points will be returned.")
            min_val = min(vals)
            for mpi in [i for i in range(len(vals)) if abs(vals[i] - min(vals)) < float_prec]:
                ok_ps.append(min_p_lst[mpi])

            print(Q)
            return Q, ok_ps
    
    elif type == 'triu':
        if len(min_p_lst) == 1:
            min_p_one_indexes = [i for i,b in enumerate(min_p_lst[0]) if b == 1]
            n_selected_point_at_m = len(min_p_one_indexes)

            Q = torch.triu((torch.rand(n,n)+1 + float_prec * 10)*coeff  * min_val)  #'''/ n_selected_point_at_m'''
            denom = (n_selected_point_at_m * (n_selected_point_at_m + 1)) // 2
            missing_cost = min_val
            for i in range(0, n_selected_point_at_m):
                for j in range(i, n_selected_point_at_m):
                    val = random.random()*missing_cost
                    missing_cost -= val
                    Q[min_p_one_indexes[i],min_p_one_indexes[j]] = val

            Q[min_p_one_indexes[n_selected_point_at_m-1], min_p_one_indexes[n_selected_point_at_m-1]] += missing_cost
            return Q, min_p_lst
        else:
            print("SORRY, not working...")
            exit()
            print("WARNING: YOU'RE TRYING TO GENERATE Q UPPER TRIANGULAR WITH MORE THAN ONE GLOBAL MINIMUM:\nthis function may reduce the number of gloabal minimum point. It also need to check all the candidate minimum point")
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
            q, _, _, _ = result

            # Reconstruct the upper triangular matrix Q from the solution vector q
            Q = torch.zeros((n, n))
            for idx, (i, j) in enumerate(upper_indices):
                Q[i, j] = q[idx]
        
            ok_ = True
            vals = []
            ok_ps = []
            for mp in min_p_lst:
                r = torch.matmul(mp, torch.matmul(Q, mp)).item()
                vals.append(r)
                if abs(r  - min_val) > float_prec:
                    ok_ = False
            if not ok_:
                print("The eq system needed to determine Q was solved unprecisely. An updated list of global min points will be returned.")
            min_val = min(vals)
            for mpi in [i for i in range(len(vals)) if abs(vals[i] - min(vals)) < float_prec]:
                ok_ps.append(min_p_lst[mpi])

            return Q, ok_ps
    


    