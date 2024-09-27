import Qcomposer as qc
import bilinear as bl
import Acomposer as ac
import torch
import solver as sl
import AL as al

torch.set_printoptions(linewidth=500)


n = 30
m = 5
n_selected_points = 10
n_min = 1
b = 6
q_type = 'diag'
vis_prob = 0.3
rem_prob = 0.1
min_val = 10

init_mu = 0.1
init_lambda = (torch.rand(m) - 0.5) / 10
init_rho = 1.5

min_ps = [bl.generate_random_binary_vector(n, n_selected_points)]

A, col_equiv = ac.gen_A(m, n, min_ps[0], b, vis_prob, rem_prob)
Q, new_min_ps = qc.gen_Q(n, q_type, min_ps, min_val)

print(f'a priori min_p ones indexes, val, bounds:')
for mp in new_min_ps:
    print(f'{[i for i, b in enumerate(mp) if b==1]}, {min_val:.2f}, {torch.matmul(A, mp).tolist()}')
print("\n")

opt, val = sl.solve_bqp(Q, A, b)
print(f'a posteriori min_p ones indexes, val, bounds:\n{[i for i, b in enumerate(list(opt)) if b == 1]}, {val:.2f}, {torch.matmul(A, torch.tensor(opt, dtype=torch.float32)).tolist()}')
print("\n")

al_min_p, mu, lambda_, iter = al.al(Q, A, b, init_mu, init_lambda, init_rho)
al_val = torch.matmul(al_min_p, torch.matmul(Q, al_min_p)).item()
print(f'AL min_p ones indexes, val, bounds:\n{[i for i, b in enumerate(al_min_p) if b==1]}, {al_val:.2f}, {torch.matmul(A, al_min_p).tolist()}')
print("\n")


if torch.equal(min_ps[0], al_min_p):
    print(f"OK! same point found in {iter+1} iterations.")
elif val == al_val:
    print(f"OK! another global minimum with value {min_val} found in {iter+1}")
else:
    print(f'ERROR: AL found a global minimum with value {al_val} in {iter+1}')

print(f'Final params: mu = {mu:.4f}, lambda = {lambda_.tolist()}')



