import bilinear as bl
import torch
import Acomposer as ac
import solver as sl
import random
import itertools
import math
import matplotlib.pyplot as plt

def max_cost(n_selected_point_at_m, max_val):
    return  max_val * n_selected_point_at_m + math.comb(n_selected_point_at_m, 2)

def histogram_from_dict(A, Q, bin_range, n_selected_point_at_min, hamming_distance_to_investigate, plot):
    """
    Plot a histogram based on the given sorted dictionary and bin size.

    Parameters:
    data (dict): A sorted dictionary with floating point keys (bin starts) and integer values (bin heights).
    bin_range (float): The size of each bin.
    """
    
    m = A.size()[0]
    n = A.size()[1]

    vals = {}
    total_exam_res = 0
    
    for k in range(n_selected_point_at_min, n_selected_point_at_min + hamming_distance_to_investigate + 1):
        # Generate all combinations of `k` positions where the vector has 1s
        for ones_positions in itertools.combinations(range(n), k):
            vec_tensor = torch.zeros(n, dtype=torch.float32)
            vec_tensor[list(ones_positions)] = 1
            # Check the condition on matrix multiplication
            if (torch.matmul(A, vec_tensor) >= b).sum().item() == m:
                # Compute the result and round down to nearest even integer
                res = torch.matmul(vec_tensor, torch.matmul(Q, vec_tensor)).item()
                res = res // 2 * 2
                
                # Update the count in the dictionary
                vals[res] = vals.get(res, 0) + 1
                total_exam_res += 1
    
    # Sort the dictionary by keys
    vals = dict(sorted(vals.items()))
    

    if plot:
        # Extract keys (bin start points) and values (heights)
        bin_starts = list(vals.keys())
        bin_heights = list(vals.values())


        # Plot the histogram
        plt.bar(bin_starts, bin_heights, width=bin_range, align='edge', edgecolor='black')

        # Labeling the plot
        plt.xlabel('Bins')
        plt.ylabel('Values (Heights)')
        plt.title('Histogram Plot from Dictionary')

        # Display the plot
        plt.show()
    return vals, total_exam_res

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
    
    print(alt_dict)
 
    new = [min_p]
    for k, v in alt_dict.items():
        if alt == 'all' or len(new) < alt + 1:
            poss = list(itertools.combinations(list(k) + v, len(v)))

            for vec in new:
                news = []
                for p in poss:
                    local = vec.clone()
                    local[v] = 0
                    # print(p)
                    local[list(p)] = 1
                    news.append(local)

            new += news
        else:
            # for n in new:
            #     print(n.tolist())
            return new[1:]
        
    return new[1:]


torch.set_printoptions(linewidth=500)

n = 20
m = 10
n_selected_point_at_m = 12
b = 8  
vis_prob = 1
rem_prob = 0.5
coeff = 1
plot_histogram = True

min_p = bl.generate_random_binary_vector(n, n_selected_point_at_m)
min_p_one_indexes = [i for i, b in enumerate(min_p.tolist()) if b == 1]
A, col_equiv = ac.gen_A(m, n, min_p, b, vis_prob, rem_prob)

print(col_equiv)

Q = torch.triu((torch.rand(n,n)+1)*coeff)
for i in range(0, n_selected_point_at_m):
    for j in range(i, n_selected_point_at_m):
        Q[min_p_one_indexes[i],min_p_one_indexes[j]] = 1
print(f'b = {b}')
print(Q)
print(A)
print(ac.visibility_counter(A))



print(f'a priori min_p ones indexes:\t\t{min_p_one_indexes} = {torch.matmul(min_p, torch.matmul(Q, min_p)):.2f}\tA*min_p = {torch.matmul(A, min_p)}')

opt, val = sl.solve_bqp(Q, A, b)
opt = torch.tensor(opt, dtype=torch.float32)
print(f'a posteriori min_p ones indexes:\t{[i for i, b in enumerate(list(opt)) if b == 1]} = {torch.matmul(opt, torch.matmul(Q, opt)):.2f}\tA*opt   = {torch.matmul(A, opt)}')

bin_range = 2

if plot_histogram:
    # Plot the histogram
    vals, n_ex_points = histogram_from_dict(A, Q,  bin_range, n_selected_point_at_m, 0, False)
    print(vals)