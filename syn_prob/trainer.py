import bilinear as bl
import torch
import torch.nn as nn
import torch.optim as optim
import random
import itertools
import statistics



def trainOne(nSelectedPointAtMinimum, n, tFs, MVal, mVal, overlap, batchSize, Q_init_type, verbose, mse_loss_adj_factor, constraint_loss_adj_factor, min_loss_adj_factor, mean_std_loss_adj_factor, randomMeanInit, randomStdInit, printEpochs, epochs, forcedMinValQUT, printFinalQs, check, b):
    """
    Trains a bilinear layer on a set of random binary vectors to learn a matrix Q that satisfies multiple constraints 
    and objectives, such as minimizing the loss, enforcing an upper triangular structure, and finding a global minimum at a specific point.

    Args:
        nSelectedPointAtMinimum (int): Number of ones in the selected minimum point vector.
        n (int): Dimensionality of the problem, i.e., the size of binary vectors.
        tFs (int): Total number of vectors in the F set.
        tsPerc (float): Percentage of vectors from F to be used as the test set.
        MVal (float): Maximum value assigned to points in the map.
        mVal (float): Value assigned to the global minimum point.
        overlap (float): Amount of overlap between vectors in the training set.
        batchSize (int): Batch size used in training.
        Q_init_type (str): Initialization strategy for the matrix Q ('identity', 'psd', 'm_biased', 'random', etc.).
        verbose (bool): If True, prints detailed information during training.
        mse_loss_adj_factor (float): Adjustment factor for MSE loss.
        constraint_loss_adj_factor (float): Adjustment factor for constraint loss.
        min_loss_adj_factor (float): Adjustment factor for global minimum loss.
        mean_std_loss_adj_factor (float): Adjustment factor for mean-std loss.
        randomMeanInit (float): Mean value used for random initialization of Q.
        randomStdInit (float): Standard deviation used for random initialization of Q.
        printEpochs (int): Frequency of printing training progress (every `printEpochs` epochs).
        epochs (int): Number of epochs to train.
        forcedMinValQUT (float): Minimum value forced in the upper triangular part of Q.
        printFinalQs (bool): If True, prints the final Q matrix after training.
        check (bool): If True, checks for global minima after training by evaluating all possible binary vectors.
        b (int): Minimum number of selected points (1s) required for each vector when checking global minima.

    Returns:
        wellDone (bool): Whether the global minimum found is the same as the expected minimum.
        global_min_value (float): The global minimum value of the learned function.
        global_max_value (float): The maximum value encountered during evaluation.
        avgValue (float): The average value over all possible vectors.
        valueAtm (float): The value of the learned function at the selected minimum point m.
        count_ones_m (int): Number of 1s in the selected minimum vector m.
        count_ones_min_point (int): Number of 1s in the vector that achieved the global minimum.
        m (torch.Tensor): The selected minimum point vector.
        min_point (tuple): The binary vector that achieved the global minimum.
        num_all_vectors (int): The total number of binary vectors checked (when `check` is True).
        nBetterMinimums (int): Number of vectors with a value better than or equal to the value at m.
        num_unique_values (int): Number of unique values of the function across all vectors.
        best_Q_epoch (int): The epoch number at which the best Q was achieved.
        nZerosUpperTriangBestQ (int): Number of zeros in the upper triangular part of the best Q matrix.
        bestQmean (float): The mean value of the upper triangular part of the best Q matrix.
        bestQstd (float): The standard deviation of the upper triangular part of the best Q matrix.
        F (list): The set of random binary vectors used in training.
        best_Q (torch.Tensor): The best Q matrix (learned during training).
        m (torch.Tensor): The selected minimum point vector.
    """
    
    
    # Generate set F and map M
    m = bl.generate_random_binary_vector(n, nSelectedPointAtMinimum)
    F, aFs  = bl.generate_F_set(n, tFs, m)
    map = bl.create_map(F, MVal, mVal, m, overlap)

    batchSize_ = min(batchSize, aFs)

    if(verbose):
        print("Problem dimensionality: ", n)
        print("Size of the training set: ", aFs)
        print(f"Selected minimum point m (value {mVal:.1f}): {[int(e) for e in m.tolist()]}\t ones at indexes {[i+1 for i, x in enumerate(m.tolist()) if x == 1]}")
        print("Training set:")
        for k, v in sorted(map.items(), key=lambda x: x[1]):
            print(f'{list(k)}, {v:.2f}\thamming distance: {bl.hamming_distance(torch.tensor(k), m)}')
        print("\n############################################################################################################\n")


    # Initialize bilinear layer
    bilinear_layer = bl.BilinearLayer(n, verbose, forcedMinValQUT,  init_type=Q_init_type, m=m, randomMeanInit=randomMeanInit, randomStdInit=randomStdInit )

    # Optimizer (e.g., Adam)
    optimizer = optim.Adam(bilinear_layer.parameters(), lr=0.01)

    # Generate random binary vectors for F
    F_vectors = [torch.tensor(f, dtype=torch.float32) for f in F]

    # Create training set (F) and test set (subset of F)
    train_set = F_vectors

    # Convert training data to pairs (x1, x2) and targets (based on the map M)
    train_pairs = [(x1, x2, map[tuple(x1.int().tolist())] + map[tuple(x2.int().tolist())])
                for x1 in train_set for x2 in train_set]

    # Training loop
    currentEpoch = 0
    ok = True
    min_total_loss = float('inf')
    best_Q = None
    best_Q_epoch = 0


    while ok:
        total_loss = 0
        total_mse_loss = 0 
        total_constraint_loss = 0
        total_min_loss = 0
        total_mean_std_loss = 0
        Q = None

        random.shuffle(train_pairs)  # Shuffle training pairs for each epoch
        
        #batch training
        for i in range(0, len(train_pairs), batchSize_):
            # Prepare a batch of input pairs
            batch = train_pairs[i:i + batchSize_]
            
            if len(batch) < batchSize_:
                continue  # Skip if the batch is smaller than the specified batch size
            
            x1_batch = torch.stack([pair[0] for pair in batch])  # Batch of x1 vectors
            x2_batch = torch.stack([pair[1] for pair in batch])  # Batch of x2 vectors
            targets = torch.tensor([pair[2] for pair in batch], dtype=torch.float32)  # Batch of targets

            optimizer.zero_grad()

            # Forward pass for the entire batch
            output = bilinear_layer(x1_batch, x2_batch).squeeze()  # Batch output (size: batchSize)

            # Force upper triangular constraint on the weight matrix Q
            Q = bilinear_layer.bilinear.weight

            # Loss based on the difference from the target values
            mse_loss = nn.functional.l1_loss(output, targets) * mse_loss_adj_factor
            constraint_loss = bl.custom_loss(Q, n) * constraint_loss_adj_factor
            min_loss = bl.global_minimum_loss(Q, m.clone().detach().requires_grad_(True)) * min_loss_adj_factor
            mean_std_loss = bl.avgStdLoss(Q, randomMeanInit, randomStdInit) * mean_std_loss_adj_factor

            # Total loss: MSE loss + constraint loss
            loss = mse_loss + constraint_loss + min_loss + mean_std_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_mse_loss += mse_loss.item()
            total_constraint_loss += constraint_loss.item()
            total_min_loss += min_loss.item()
            total_mean_std_loss += mean_std_loss.item()

        # Print progress
        if (currentEpoch + 1) % printEpochs == 0 and verbose:
            print(f'Epoch {currentEpoch+1}/{epochs}, Loss: {total_mse_loss:.4f} + {total_constraint_loss:.4f} + {total_min_loss:.4f} + {total_mean_std_loss:.4f} = {total_loss:.4f}')

        currentEpoch += 1
        ok = currentEpoch < epochs 
        if not ok and (currentEpoch + 1) % printEpochs != 0 and verbose:
            print(f'Epoch {currentEpoch+1}/{epochs}, Loss: {total_mse_loss:.1f}\t{total_constraint_loss:.1f}\t{total_min_loss:.1f} + {total_mean_std_loss:.4f} = {total_loss:.4f}')

        if total_loss < min_total_loss:
            min_total_loss = total_loss
            best_Q = Q.clone()  # Save a copy of the best Q
            best_Q_epoch = currentEpoch

    # Extract the learned matrix Q (from the bilinear layer)
    #Q = bilinear_layer.bilinear.weight.squeeze()
    Q = best_Q.squeeze()

    if(verbose):
        print("\n############################################################################################################\n")


    # Print the matrix Q
    if(verbose):
        print("Best learned matrix Q at epoch ", best_Q_epoch)
        print(Q)

    Q = Q.detach()

    bl.makeQUpperTriangular(Q, forcedMinValQUT)
    #makeQPositiveValued(Q)

    if(printFinalQs):
        print("\nZeroed Q:")
        print(Q)

        print("\n############################################################################################################\n")


    # Compute m^T * Q * m (the target minimum value)
    min_value_at_m = torch.matmul(m, torch.matmul(Q, m)).item()
    if(verbose):
        print(f"Value at m (global minimum candidate): {min_value_at_m}")

    nBetterMinimums = -1
    global_min_value = float('-inf')
    global_max_value = float('-inf')        
    avgValue = float('-inf')
    min_point = torch.tensor([-1 for _ in range(0, n)])
    values = set([])
    res = {} 
    wellDone = -1
    all_vectors = []


    if check:
        # Generate all possible binary vectors of length n and removing 0 (it is not a feasible point whatever A is)
        all_vectors = [torch.tensor(list(vec), dtype=torch.float32) for vec in itertools.product([0, 1], repeat=n) if vec.count(1) >= b]

        # Variable to track if m has the global minimum
        global_min_value = float('inf')
        global_max_value = float('-inf')        #not needed, just to test statistics
        avgValue = 0
        min_point = None
        

        # Check all possible binary vectors
        for x_tensor in all_vectors:
            # Compute x^T * Q * x
            value = torch.matmul(x_tensor, torch.matmul(Q, x_tensor)).item()
            values.add(value)

            res[tuple(x_tensor.tolist())] = value


            # Track the minimum value and the corresponding vector
            if value < global_min_value:
                global_min_value = value
                min_point = x_tensor.detach()
            if value > global_max_value:
                global_max_value = value
            avgValue += value / len(all_vectors)
        
        # Print the value at each vector
        if(verbose):
            counter = 0
            for k, v in sorted(res.items(), key=lambda x: x[1]):
                if counter >= 50:
                    break
                print(f"Value at x = {[int(xi) for xi in list(k)]}: {v:.2f}\t#hamming = {bl.hamming_distance(torch.tensor(k), m)}\t#selected points = {k.count(1   )}")
                counter += 1

        # Final check to see if the global minimum is at m
        valueAtm = torch.matmul(m, torch.matmul(Q, m)).item()
        wellDone = global_min_value == valueAtm
        if(verbose):
            if wellDone:
                print(f"OK: The global minimum has the same value to the function in m: {global_min_value} ")
            else:
                print(f"[!] : The global minimum has different value wrt the function in m: {global_min_value} instead of {valueAtm}")

        nBetterMinimums = sorted(values).index(valueAtm)
    
    valueAtm = torch.matmul(m, torch.matmul(Q, m)).item()
    nZerosUpperTriangBestQ = bl.custom_loss(best_Q, n)

    

    if(verbose):
        print("\n############################################################################################################\n")
   
    triuQ = []
    for i in range(0, n):
        for j in range(i, n):
            triuQ.append(Q[i][j].item())


    bestQmean = statistics.mean(triuQ)
    bestQstd = statistics.stdev(triuQ)

    return wellDone, global_min_value, global_max_value, avgValue, valueAtm, m.tolist().count(1), min_point.tolist().count(1), m, tuple(min_point.tolist()), len(all_vectors), nBetterMinimums, len(values), best_Q_epoch, nZerosUpperTriangBestQ, bestQmean, bestQstd, F, best_Q, m
