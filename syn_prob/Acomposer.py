import torch
import random
import math
import itertools

# def composeA(m, n, minPoint, b, visProb, onePer):
#     """
#     Composes a binary matrix `A` of size (m, n) such that minPoint is a feasible point and at least onePer of the entries are set to one.
    
#     Args:
#         m (int): The number of rows in the matrix `A`.
#         n (int): The number of columns in the matrix `A`.
#         minPoint (list): A binary vector of size `n` representing the minimum point, where 1s indicate selected columns.
#         b (int): The number of guaranteed 1s in each row, randomly selected from the columns indicated by 1 in `minPoint`.
#         visProb (float): Probability of adding additional 1s in the columns indicated by `minPoint` but not selected initially.
#         onePer (float): The desired proportion of 1s in the entire matrix `A`.

#     Returns:
#         A (torch.Tensor): A binary matrix of size (m, n), where each row has exactly `b` guaranteed 1s and additional 1s are 
#                           placed with probability `visProb`. The total number of 1s in the matrix is approximately `m * n * onePer`.
#     """


#     A = torch.zeros(m, n, dtype = torch.float32)
#     oneIndexes = [i for i, x in enumerate(list(minPoint)) if x == 1]
#     global_missing = oneIndexes.copy()
#     freePos = list(range(0, m*n))
#     vis_added_indexes = []

#     for r in range(0,m):
#         missing = oneIndexes.copy()
#         nTaken = min(len(global_missing), b)
#         for c in random.sample(global_missing, nTaken):
#             A[r][c] = 1
#             global_missing.remove(c)
#             missing.remove(c)
#             freePos.remove(r*n + c)

#         for c in random.sample(missing, b - nTaken):
#             A[r][c] = 1
#             missing.remove(c)
#             freePos.remove(r*n + c)

#         # print(" ", A[r])

#     blocked = torch.tensor([False for _ in range(0,m)])
#     new_A = A.clone()

#     randomOneIndexes = oneIndexes.copy()
#     random.shuffle(randomOneIndexes)
#     already_added = torch.tensor([False for _ in range(0,m)])
#     for c in randomOneIndexes:
#         randomRows = list(range(0,m))
#         random.shuffle(randomRows)
#         for r in randomRows:
#             if A[r][c] == 0 and random.random() < visProb:
#                 toAdd = torch.zeros(m)
#                 toAdd[r] = 1
#                 ok = True
#                 for c1 in oneIndexes:
#                     if c1 != c and torch.logical_and(torch.logical_or(blocked, toAdd), A[:,c1]).sum().item() >= A[:,c1].sum().item():
#                         ok = False
#                         break
#                 if ok:
#                     new_A[r,c] = 1
#                     blocked[r] = True

#     # nOnes = (A == 1).sum().item()
#     # if nOnes < int(m*n*onePer - nOnes):
#     #     for compInd in random.sample(freePos, int(m*n*onePer - nOnes)):
#     #         A[compInd // n][compInd % n] = 1


#     return new_A

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

# def add_visibility(A, level, n_point_to_add, min_p, b):
    
#     m = A.size()[0]
#     n = A.size()[1]

#     vis_map = point_locator(A)

#     npa = n_point_to_add    
#     if npa == 'all':
#         npa = len([c for c in range(0, n) if (len(vis_map[c]) < level and min_p[c] == 0)])

#     added = 0
#     up_level = 0

#     forbidden_rows_idxs = [i for i, b in enumerate((torch.matmul(A, min_p) == b)) if b == True]
#     # print(f'forbidden_cols_idxs: {forbidden_cols_idxs}')
#     while added < npa:
#         free_index = [c for c in range(0, n) if (len(vis_map[c]) == up_level and min_p[c] == 0)]
#         # print(free_index)
#         column_to_fill = random.sample(free_index, min(len(free_index), npa-added))
#         # print(column_to_fill)
#         added += len(column_to_fill)
#         # print(added)
#         for c in column_to_fill:
#             n_improvable = len(forbidden_rows_idxs)

#             available_pos = [i for i in range(0,m) if i not in vis_map[c]]
#             try:
#                 to_remove = random.sample(forbidden_rows_idxs, len(forbidden_rows_idxs)-1)
#                 for r in to_remove:
#                     available_pos.remove(r)
#             except ValueError:
#                 pass
#             # print(f'available_pos before: {available_pos}')
#             removable = [i for i in available_pos if i in forbidden_rows_idxs]
#             # print(f'removable: {removable}')
#             # try:
#             #     available_pos.remove(random.sample(removable, 1)[0])
#             # except ValueError:
#             #     pass
#             # print(f'available_pos: {available_pos}')
#             if len(available_pos) >= level - len(vis_map[c]):
#                 rows = random.sample(available_pos, level - len(vis_map[c]))
#                 # for i in rows:
#                 #     if i in forbidden_rows_idxs
#                 A[rows, c] = 1
                
#         up_level += 1

#     return A

def gen_A(m, n, min_p, b, vis_prob, rem_prob):
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

    same_cols = {}
    #detecting same columns                                                             
    for left_c in range(0, n_selected_c):
        for right_c in range(0, n-n_selected_c):
            if (A[:,selected_idx[left_c]] == A[:,not_selected_idx[right_c]]).sum().item() == m:
                try:
                    same_cols[selected_idx[left_c]].append(not_selected_idx[right_c])
                except KeyError:
                    same_cols[selected_idx[left_c]] = [not_selected_idx[right_c]]
    
    return A, same_cols

            


