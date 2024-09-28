import Qcomposer as qc
import bilinear as bl
import Acomposer as ac
import torch
import solver as sl
import AL as al

torch.set_printoptions(linewidth=500)


n = 100
m = 8
n_selected_points = 10
n_min = 1
b = 7
q_type = 'triu'
vis_prob = 0.3
rem_prob = 0.1
min_val = 10

init_mu = 0.1
init_lambda = (torch.rand(m) - 0.5) / 10
init_rho = 2
float_prec = 1e-5

print(f'n = {n}, m = {m}, b = {b}')

print("Generating global minimum...")
min_ps = [bl.generate_random_binary_vector(n, n_selected_points)]

print("Generating A...")
A, _ = ac.gen_A(m, n, min_ps[0], b, vis_prob, rem_prob)
print("Generating Q...")
Q, new_min_ps = qc.gen_Q(n, q_type, min_ps, min_val)


print(f'A priori global minimum one indexes, val, bounds:')
print(f'{[i for i, b in enumerate(min_ps[0]) if b==1]}, {min_val:.2f}, {torch.matmul(A, min_ps[0]).tolist()}')
print("\n")


print("Solving with bounds")
opt, val = sl.solve_bqp(Q, A, b)
if abs(val - min_val) > float_prec:
    print(f'ERROR: Q or A are wrong: the global minimum value was supposed to be {min_val}, insted {val} was found.')
    exit()
elif not torch.equal(min_ps[0], torch.tensor(opt, dtype=torch.float32)):
    print(f'This objective function has more than one global minimum')

# print("Solving with AL with CPLEX")
# Q_cplex = Q.clone().float()
# for i in range(0, n):
#     for j in range(0, i):
#         Q_cplex[i, j] = Q_cplex[j, i]
# al_cplex_min_p, mu_cplex, lambda_cplex, iter_cplex = al.al_cplex(Q_cplex, A, b, init_mu, init_lambda, init_rho)
# al_cplex_val = torch.matmul(al_cplex_min_p, torch.matmul(Q, al_cplex_min_p)).item()

print("Solving with AL with QBsolv")
al_qb_min_p, mu_qb, lambda_qb, iter_qb = al.al_qbsolv(Q, A, b, init_mu, init_lambda, init_rho)
al_qb_val = torch.matmul(al_qb_min_p, torch.matmul(Q, al_qb_min_p)).item()

print("Solving with AL with simulated-annealing")
al_dimod_min_p, mu_dimod, lambda_dimod, iter_dimod = al.al_dimod(Q, A, b, init_mu, init_lambda, init_rho)
al_dimod_val = torch.matmul(al_dimod_min_p, torch.matmul(Q, al_dimod_min_p)).item()



print("Global minimum one indexes, val, bounds:")
print(f'{[i for i, b in enumerate(list(opt)) if b == 1]}, {val:.2f}, {torch.matmul(A, torch.tensor(opt, dtype=torch.float32)).tolist()}')
print("\n")



print("CPLEX - AL global minimum one indexes, val, bounds:")
print(f'{[i for i, b in enumerate(al_cplex_min_p) if b==1]}, {al_cplex_val:.2f}, {torch.matmul(A, al_cplex_min_p).tolist()}')
print("\n")

# if torch.equal(min_ps[0], al_cplex_min_p):
#     print(f"OK! same point found in {iter_cplex+1} iterations.")
# elif abs(val - al_cplex_val) < float_prec:
#     print(f"OK! another global minimum with value {min_val} found in {iter_cplex+1}")
# else:
#     print(f'ERROR: AL found a global minimum with value {al_cplex_val} in {iter_cplex+1}')

# print(f'Final params: mu = {mu_cplex:.4f}, lambda = {lambda_cplex.tolist()}\n')



print("QBsolver-AL global minimum one indexes, val, bounds:")
print(f'{[i for i, b in enumerate(al_qb_min_p) if b==1]}, {al_qb_val:.2f}, {torch.matmul(A, al_qb_min_p).tolist()}')
print("\n")

if torch.equal(min_ps[0], al_qb_min_p):
    print(f"OK! same point found in {iter_qb+1} iterations.")
elif abs(val - al_qb_val) < float_prec:
    print(f"OK! another global minimum with value {min_val} found in {iter_qb+1}")
else:
    print(f'ERROR: AL found a global minimum with value {al_qb_val} in {iter_qb+1}')

print(f'Final params: mu = {mu_qb:.4f}, lambda = {lambda_qb.tolist()}')



print("Simulated-Annealer-AL global minimum one indexes, val, bounds:")
print(f'{[i for i, b in enumerate(al_dimod_min_p) if b==1]}, {al_dimod_val:.2f}, {torch.matmul(A, al_dimod_min_p).tolist()}')
print("\n")

if torch.equal(min_ps[0], al_dimod_min_p):
    print(f"OK! same point found in {iter_dimod+1} iterations.")
elif abs(val - al_dimod_val) < float_prec:
    print(f"OK! another global minimum with value {min_val} found in {iter_dimod+1}")
else:
    print(f'ERROR: AL found a global minimum with value {al_dimod_val} in {iter_dimod+1}')

print(f'Final params: mu = {mu_dimod:.4f}, lambda = {lambda_dimod.tolist()}')