import bilinear as bl
import torch
import Acomposer as ac
import solver as sl
import random
import itertools

torch.set_printoptions(linewidth=500)

n = 20
m = 10
n_selected_point_at_m = 7
b = 4
vis_prob = 0
one_per = 0

min_p = bl.generate_random_binary_vector(n, n_selected_point_at_m)
min_p_one_indexes = [i for i, b in enumerate(min_p.tolist()) if b == 1]
A = ac.gen_A(m, n, min_p, b, vis_prob, 0.3)

Q = torch.zeros(n, n)
for d in range(0,n):
    if d+1 in min_p_one_indexes:
        Q[d,d] = random.random()
    else:
        Q[d,d] = (random.random() + 1) * n_selected_point_at_m

print(A)
print(ac.visibility_counter(A))

print(f'a priori min_p ones indexes:\t\t{min_p_one_indexes} = {torch.matmul(min_p, torch.matmul(Q, min_p)):.2f}\tA*min_p = {torch.matmul(A, min_p)}')

opt, val = sl.solve_bqp(Q, A, b)
print(f'a posteriori min_p ones indexes:\t{[i for i, b in enumerate(list(opt)) if b == 1]} = {val:.2f}\tA*opt = {torch.matmul(A, torch.tensor(opt, dtype=torch.float32))}')


