import torch
import random
import math
import itertools

def point_locator(A):
    map = {}
    m = A.size()[0]
    n = A.size()[1]

    for c in range(0, n):
        vis_list = []
        for r in range(0,m):
            if A[r][c] == 1:
                vis_list.append(r)
            map[c] = vis_list

    return map

def visibility_counter(A):
    vis = point_locator(A)
    counter = {}
    for k in vis:
        try:
            counter[len(vis[k])] += 1
        except KeyError:
            counter[len(vis[k])] = 1

    return dict(sorted(counter.items()))

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


def gen_A(m, n, min_p, b, vis_prob, rem_prob, check_equal_cols = False):
    n_selected_c = int(min_p.sum().item())
    selected_idx = [i for i, b in enumerate(min_p.tolist()) if b == 1]
    not_selected_idx = [i for i, b in enumerate(min_p.tolist()) if b == 0]
    diag_len = math.ceil(n_selected_c / b)


    up_left_A = torch.zeros(diag_len, diag_len*b)
    in_more_than_one_image_points = []
    if m < diag_len or n < n_selected_c:
        raise ValueError("m or n to small")
    for i in range(0, b):
        up_left_A[:, i*diag_len:(i+1)*diag_len] = torch.eye(diag_len)
    n_point_to_add = up_left_A[:, n_selected_c:].sum(1)
    up_left_A = up_left_A[:, 0:n_selected_c]

    for r in range(0, diag_len):
        cols = random.sample([i for i in range(0, n_selected_c) if i % diag_len != r], int(n_point_to_add[r].item()))
        up_left_A[r, cols] = 1
        in_more_than_one_image_points += cols

    # print(f'upa:\n{up_left_A}')
    # print(in_more_than_one_image_points)


    up_right_A = torch.zeros(diag_len, n-n_selected_c)
    single_vis_points = [i for i in range(0, n_selected_c) if i not in in_more_than_one_image_points]
    # print(single_vis_points)
    for c in range(0, n-n_selected_c):
        up_right_A[:, c] = up_left_A[:, random.sample(single_vis_points, 1)[0]]
    up_right_A = torch.logical_and(up_right_A, torch.rand(diag_len, n-n_selected_c) >= rem_prob).float()
    #up_right_A = torch.zeros(diag_len, n-n_selected_c)


    low_left_A = torch.zeros(m-diag_len, n_selected_c)
    for r in range(0, m-diag_len):
        low_left_A[r, random.sample(list(range(0, n_selected_c)), random.randint(b, n_selected_c-1))] = 1
        #low_left_A[:,:] = 1

    # low_left_A = (torch.rand(m-diag_len, n_selected_c) < vis_prob).float()
    low_right_A = (torch.rand(m-diag_len, n-n_selected_c) < vis_prob).float()
    #low_right_A = torch.zeros(m-diag_len, n-n_selected_c)


    left_A = torch.cat((up_left_A, low_left_A), 0)
    # # print(f'lA = \n{left_A}')                    
    # left_A = torch.transpose(left_A, 0, 1)              #shuffling columns
    # left_A_l = left_A.tolist()
    # random.shuffle(left_A_l)
    # left_A = torch.tensor(left_A_l)
    # left_A = torch.transpose(left_A, 0, 1)


    right_A = torch.cat((up_right_A, low_right_A), 0)
    # right_A = torch.zeros(m, n-n_selected_c)
    # print(f'rA =\n{right_A}')


    A = torch.zeros(m, n)

    for i, c in enumerate(selected_idx):
        A[:, c] = left_A[:, i]

    for i, c in enumerate(not_selected_idx):
        A[:, c] = right_A[:, i]

    # A_l = A.tolist()                    #shuffling rows
    # random.shuffle(A_l)
    # A = torch.tensor(A_l)

    #check no empty columns and eventually fix it
    for c in range(0,n):
        if A[:, c].sum().item() == 0:
            A[random.randint(diag_len, m-1), c] = 1

    equal_cols = {}
    if check_equal_cols:
        #detecting same columns                                                             
        for left_c in range(0, n_selected_c):
            for right_c in range(0, n-n_selected_c):
                if (A[:,selected_idx[left_c]] == A[:,not_selected_idx[right_c]]).sum().item() == m:
                    try:
                        equal_cols[selected_idx[left_c]].append(not_selected_idx[right_c])
                    except KeyError:
                        equal_cols[selected_idx[left_c]] = [not_selected_idx[right_c]]
    
    return A, equal_cols

            


