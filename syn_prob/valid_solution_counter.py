import bilinear as bl
import torch
import Acomposer as ac
import solver as sl
import random
import itertools

def generate_alternative_vectors(min_p, swap_dict, alt):
    """
    Generate alternative vectors by swapping bits according to a dictionary of swap indices.
    
    Parameters:
    min_p (list): The original binary vector.
    swap_dict (dict): A dictionary where the keys are indices of min_p and values are lists of possible swap indices.
    alt (int): The maximum number of alternative vectors to generate.
    
    Returns:
    list: A list of alternative vectors.
    """


    alt_dict = {}
    for k, v in swap_dict.items():
        try:
            alt_dict[tuple(v)].append(k)
        except KeyError:
            alt_dict[tuple(v)] = [k]
     
    new = [min_p]

    for k, v in alt_dict.items():
        poss = list(itertools.combinations(list(k) + v, len(v)))
        poss.remove(tuple(v))
        # print(f'k: {k}, v: {v}')            
        vec_i = 0
        news = []
        for vec in new:
            # print(new)
            for p in poss:
                local = vec.clone()
                local[v] = 0
                # print(p)
                local[list(p)] = 1
                for t in news:
                    if torch.equal(t, local):
                        print(f"something strange here: p:{p}, local:{local}, old:{t}")
                news.append(local)
                # print(local)
        new += news
        
        if alt != 'all' and len(new) >= alt:
            return new[1:alt+1]
        
    return new[1:]
    


torch.set_printoptions(linewidth=500)

n = 15
m = 5
n_selected_point_at_m = 7
b = 4
vis_prob = 0.7  
rem_prob = 0.3


min_p = bl.generate_random_binary_vector(n, n_selected_point_at_m)
min_p_one_indexes = [i+1 for i, b in enumerate(min_p.tolist()) if b == 1]
A, sd = ac.gen_A(m, n, min_p, b, vis_prob, rem_prob)


all_vectors = [torch.tensor(list(vec), dtype=torch.float32) for vec in itertools.product([0, 1], repeat=n)] #if vec.count(1) >= b]
valid_solution_counter = 0

print("Start searching...")
vals = {}
for i in range(0,n+1):
    vals[i] = 0

for vec in all_vectors:
    cond = torch.matmul(A, vec)
    # print(vec, cond, sep = "\t")
    if (torch.matmul(A, vec) >= b).sum().item() == m:
        val = (vec == 1).sum().item()
        vals[val] += 1
        valid_solution_counter += 1
              
print(A)
total_differt_values = 0

for i in range(0,n+1):
    print(f'{i} cost: {vals[i]}')

s = 'all'
print(f'number of valid solution = {valid_solution_counter}')
eq = generate_alternative_vectors(min_p, sd, 100000)
print(len(eq))


print(A)
