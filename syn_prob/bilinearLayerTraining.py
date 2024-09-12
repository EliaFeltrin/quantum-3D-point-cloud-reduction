import torch
import torch.nn as nn
import torch.optim as optim
import random
import itertools

def generate_random_binary_vector(n):
    # Generates a binary vector of length n
    return torch.randint(0, 2, (n,), dtype=torch.int8)

def generate_F_set(n, l):
    # Creates a set of random binary vectors of length n
    F = {tuple(generate_random_binary_vector(n).tolist()) for _ in range(l)}
    return set(F), len(F)

def create_map(F, MVal):
    # Select a random element m from F
    tmpLst = list(F).copy() 
    random.shuffle(tmpLst)
    m = None
    for p in tmpLst:
        if(p.count(1) >= b):
            m = p
            break
    
    # Initialize the map M
    M = {}
    
    for f in F:
        if f == m:
            M[f] = mVal  # Map m to the lowest value
        else:
            M[f] = random.uniform(mVal, MVal)  # Map other vectors to a random positive value
    
    return M, m

# Define a bilinear layer without bias
class BilinearLayer(nn.Module):
    def __init__(self, n, init_type='identity', m=None):
        super(BilinearLayer, self).__init__()
        self.bilinear = nn.Bilinear(n, n, 1, bias=False)  # No bias

        # Initialize the weight based on the selected strategy
        self.initialize_weights(init_type, m)

    def initialize_weights(self, init_type, m):
        # Access the 3D weight tensor of the bilinear layer
        weight = self.bilinear.weight.data

        if init_type == 'id':
            # Initialize each 2D slice of the 3D weight tensor as an identity matrix
            if weight.shape[1] == weight.shape[2]:  # Ensure dimensions match for identity
                for i in range(weight.shape[0]):
                    nn.init.eye_(weight[i])  # Initialize each slice
            else:
                raise ValueError("Input dimensions must match for identity initialization")
        
        elif init_type == 'near_id':
            # Initialize as a matrix close to identity with small random noise
            if weight.shape[1] == weight.shape[2]:
                for i in range(weight.shape[0]):
                    nn.init.eye_(weight[i])  # Start with identity
                    weight[i] += torch.randn_like(weight[i]) * 0.01  # Add small noise
            else:
                raise ValueError("Input dimensions must match for near_identity initialization")
        
        elif init_type == 'psd':
            # Initialize to ensure the weight matrix is positive semi-definite
            for i in range(weight.shape[0]):
                A = torch.randn(weight.shape[1], weight.shape[2])
                weight[i] = torch.mm(A, A.t())  # A * A^T is positive semi-definite

        elif init_type == 'm_biased':
            if m is None:
                raise ValueError(f"m is None")
            # Bias the weight matrix towards m
            else:
                m_tensor = torch.tensor(m, dtype=torch.float32).unsqueeze(1)  # Shape (n, 1)
                for i in range(weight.shape[0]):
                    weight[i] = torch.mm(m_tensor, m_tensor.t())  # Outer product of m with itself

                # Optionally add a small random perturbation to avoid singularities
                weight += torch.randn_like(weight) * 0.01

        elif init_type == 'random':
            # Random initialization (PyTorch default)
            nn.init.normal_(weight, mean=0.0, std=0.01)

        else:
            raise ValueError(f"Unknown initialization type: {init_type}")

        # Register backward hook to modify the weights after each backward pass
        self.bilinear.weight.register_hook(self.enforce_upper_triangular)

    def forward(self, x1, x2):
        # Force the weight matrix to be upper triangular in the forward pass
        with torch.no_grad():
            Q = self.bilinear.weight
            Q.copy_(torch.triu(Q))  # Zero out the lower triangular part (including below diagonal)

        return self.bilinear(x1, x2)

    def enforce_upper_triangular(self, grad):
        """ Hook function that forces the weight to be upper triangular after each backward pass. """
        with torch.no_grad():
            Q = self.bilinear.weight
            Q.copy_(torch.triu(Q))  # Ensure the matrix remains upper triangular
        
        return grad  # Return the gradient unmodified (for normal backward pass)


def custom_loss(Q, n):
    # Extract the upper triangular matrix including diagonal
    upper_triangular = torch.triu(Q)
    
    # Penalize negative or zero values in the upper triangular part
    upper_loss = torch.sum(torch.relu(-upper_triangular))
    
    # Extract the lower triangular part (below the diagonal)
    lower_triangular = torch.tril(Q, diagonal=-1)
    
    # Penalize any non-zero values in the lower triangular part
    lower_loss = torch.sum(lower_triangular ** 2)  # Squaring to enforce exactly zero
    
    # Total loss: penalties for both upper and lower parts
    total_loss = upper_loss + lower_loss
    
    return total_loss

def global_minimum_loss(Q, m_tensor):
    # Compute m^T * Q * m
    m_tensor = m_tensor.unsqueeze(0)
    min_value_at_m = torch.matmul(m_tensor, torch.matmul(Q, m_tensor.T))
    
    # Penalize if the value at m is not sufficiently small (encourage Q to minimize at m)
    return torch.relu(min_value_at_m)


# Function to generate all possible binary vectors of length n but 0
def generate_all_binary_vectors(n, b):
    return [torch.tensor(list(vec), dtype=torch.float32) for vec in itertools.product([0, 1], repeat=n) if vec.count(1) >= b]

def makeQUpperTriangular(Q):
    for i in range(0,n):
        for j in range(0, i):
            Q[i,j] = 0

def makeQPositiveValued(Q):
    for i in range(0,n):
        for j in range(0,n):
            Q[i,j] = abs(Q[i,j])


def trainOne():
    # Generate set F and map M
    F, aFs  = generate_F_set(n, tFs)
    tsSize  = int(aFs * tsPerc)
    map, m = create_map(list(F), MVal)

    if(verbose):
        print("Problem dimensionality: ", n)
        print("Size of the training set: ", aFs)
        print("Size of the test set: ", tsSize)
        print(f"Selected minimum point m (value {mVal:.1f}): {m}")
        print("Training set:", map)
        print("\n############################################################################################################\n")


    # Initialize bilinear layer
    bilinear_layer = BilinearLayer(n, init_type=Q_init_type, m=m)

    # Optimizer (e.g., Adam)
    optimizer = optim.Adam(bilinear_layer.parameters(), lr=0.01)

    # Generate random binary vectors for F
    F_vectors = [torch.tensor(f, dtype=torch.float32) for f in F]

    # Create training set (F) and test set (subset of F)
    train_set = F_vectors
    test_set = random.sample(F_vectors, tsSize)

    # Convert training data to pairs (x1, x2) and targets (based on the map M)
    train_pairs = [(x1, x2, map[tuple(x1.int().tolist())] + map[tuple(x2.int().tolist())])
                for x1 in train_set for x2 in train_set]

    '''while(ok):
        total_loss = 0
        for x1, x2, target in train_pairs:
            optimizer.zero_grad()
            output = bilinear_layer(x1, x2)
            
            # Loss based on the difference from the target value
            mse_loss = nn.functional.mse_loss(output, torch.tensor([target]))
            
            # Force upper triangular constraint on the weight matrix Q
            Q = bilinear_layer.bilinear.weight
            constraint_loss = custom_loss(Q, n)
            
            # Total loss: MSE loss + constraint loss
            loss = mse_loss + constraint_loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        
        # Print progress
        if (currentEpoch + 1) % printEpochs == 0:
            print(f'Epoch {currentEpoch+1}/{epochs}, Loss: {total_loss:.4f}')

        currentEpoch += 1
        ok = currentEpoch < epochs and abs(oldLoss - total_loss) > 0.1 
        if(not ok and (currentEpoch + 1) % printEpochs != 0):
            print(f'Epoch {currentEpoch+1}/{epochs}, Loss: {total_loss:.4f}')
        oldLoss = total_loss
    '''
    # Training loop
    currentEpoch = 0
    oldLoss = float('inf')
    ok = True
    min_total_loss = float('inf')
    best_Q = None
    best_Q_epoch = 0
    while ok:
        total_loss = 0
        total_mse_loss = 0 
        total_constraint_loss = 0
        total_min_loss = 0
        Q = None
        random.shuffle(train_pairs)  # Shuffle training pairs for each epoch
        for i in range(0, len(train_pairs), batchSize):
            # Prepare a batch of input pairs
            batch = train_pairs[i:i + batchSize]
            
            if len(batch) < batchSize:
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
            mse_loss = nn.functional.mse_loss(output, targets) * mse_loss_adj_factor
            constraint_loss = custom_loss(Q, n) * constraint_loss_adj_factor
            min_loss = global_minimum_loss(Q, torch.tensor(m, dtype=torch.float32)) * min_loss_adj_factor

            # Total loss: MSE loss + constraint loss
            loss = mse_loss + constraint_loss + min_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_mse_loss += mse_loss.item()
            total_constraint_loss += constraint_loss.item()
            total_min_loss += min_loss.item()

        # Print progress
        if (currentEpoch + 1) % printEpochs == 0 and verbose:
            print(f'Epoch {currentEpoch+1}/{epochs}, Loss: {total_mse_loss:.4f} + {total_constraint_loss:.4f} + {total_min_loss:.4f} = {total_loss:.4f}')

        currentEpoch += 1
        ok = currentEpoch < epochs #and abs(oldLoss - total_loss) > 0.1
        if not ok and (currentEpoch + 1) % printEpochs != 0 and verbose:
            print(f'Epoch {currentEpoch+1}/{epochs}, Loss: {total_mse_loss:.1f}\t{total_constraint_loss:.1f}\t{total_min_loss:.1f} = {total_loss:.4f}')
        oldLoss = total_loss

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
    if(verbose or True):
        print("Best learned matrix Q at epoch ", best_Q_epoch)
        print(Q)

    Q = Q.detach()

    makeQUpperTriangular(Q)
    #makeQPositiveValued(Q)

    if(verbose):
        print("\nZeroed Q:")
        print(Q)

        print("\n############################################################################################################\n")


    # Convert the chosen minimum vector m to a tensor
    m_tensor = torch.tensor(m, dtype=torch.float32)

    # Compute m^T * Q * m (the target minimum value)
    min_value_at_m = torch.matmul(m_tensor, torch.matmul(Q, m_tensor)).item()
    if(verbose):
        print(f"Value at m (global minimum candidate): {min_value_at_m}")

    # Generate all possible binary vectors of length n and removing 0 (it is not a feasible point whatever A is)
    all_vectors = generate_all_binary_vectors(n, b)

    # Variable to track if m has the global minimum
    global_min_value = float('inf')
    global_max_value = float('-inf')        #not needed, just to test statistics
    avgValue = 0
    min_point = None
    values = set([])                             
    # Check all possible binary vectors
    for x_tensor in all_vectors:
        # Compute x^T * Q * x
        value = torch.matmul(x_tensor, torch.matmul(Q, x_tensor)).item()
        values.add(value)
        
        # Print the value at each vector
        if(verbose):
            print(f"Value at x = {[int(xi) for xi in x_tensor.tolist()]}: {value}")

        # Track the minimum value and the corresponding vector
        if value < global_min_value:
            global_min_value = value
            min_point = x_tensor.detach()
        if value > global_max_value:
            global_max_value = value
        avgValue += value / len(all_vectors)

    if(verbose):
        print("\n############################################################################################################\n")


    # Final check to see if the global minimum is at m
    valueAtm = torch.matmul(m_tensor, torch.matmul(Q, m_tensor)).item()
    wellDone = global_min_value == valueAtm
    if(verbose):
        if wellDone:
            print(f"OK: The global minimum has the same value to the function in m: {global_min_value} ")
        else:
            print(f"[!] : The global minimum has different value wrt the function in m: {global_min_value} instead of {valueAtm}")

    nBetterMinimums = sorted(values).index(valueAtm)

    return wellDone, global_min_value, global_max_value, avgValue, valueAtm, m.count(1), min_point.tolist().count(1), m, tuple(min_point.tolist()), len(all_vectors), nBetterMinimums, len(values)

def updateAvg(avg, iter, newVal):
    if(iter == 0):
        return newVal
    else:
        return (avg * iter + newVal) / (iter + 1)

# Parameters

torch.set_printoptions(linewidth=200)

n = 9                      # problem dimensionality
tFs = min(64, pow(2,n))     # tentative size of the subset of feasible point (generated as random, the duplicates are removed)
tsPerc = 0.5                # size of the test set
MVal = 100.0                 # maximum value possibly reachable by the func
mVal = 1.2
epochs = 10
printEpochs = 5000
batchSize = 64
mse_loss_adj_factor = .0
constraint_loss_adj_factor = 1.0
min_loss_adj_factor = 50.0
Q_init_type = 'near_id'          #choose between id, near_id, psd, m_biased
b = 2                       # minimum number of point that amust be visible for each image
nTest = 500

verbose = True

totalOk = 0
avgMinVal = 0
avgMaxVal = 0
avgAvgVal = 0
avgValueAtm = 0
avgNOnesIn_m = 0
avgNOnesInActualMin = 0 
avgNBetterMinimums = 0
mSet = set([])
minPointset = set([])
print("iter\t\tOK\t\tavgMinVal\tavgVal\t\tavgMaxVal\tavgValm\t\tavg #ones @m\tavg #ones @min\t# distinct m\t# distinct min point\tavg # better minimums on wrong Qs")
verbose = False
for iter in range(0, nTest):
    wellDone, global_min_value, global_max_value, avgValue, valueAtm, nOnesIn_m, nOnesInActualMin, m, minPoint, nSensiblePoints, nBetterMinimums, nDistinctValues = trainOne()
    totalOk += wellDone
    avgMinVal = updateAvg(avgMinVal, iter, global_min_value)
    avgAvgVal = updateAvg(avgAvgVal, iter, avgValue)
    avgMaxVal = updateAvg(avgMaxVal, iter, global_max_value)
    avgValueAtm = updateAvg(avgValueAtm, iter, valueAtm)
    avgNOnesIn_m = updateAvg(avgNOnesIn_m, iter, nOnesIn_m)
    avgNOnesInActualMin = updateAvg(avgNOnesInActualMin, iter, nOnesInActualMin)
    if(not wellDone):
        avgNBetterMinimums = updateAvg(avgNBetterMinimums, iter - totalOk, nBetterMinimums)
    mSet.add(m)
    minPointset.add(minPoint)

    print(f'{iter}\t\t{totalOk}/{iter+1}\t\t{avgMinVal:.2f}\t\t{avgAvgVal:.2f}\t\t{avgMaxVal:.2f}\t\t{avgValueAtm:.2f}\t\t{avgNOnesIn_m:.2f}/{n}\t\t{avgNOnesInActualMin:.2f}/{n}\t\t{len(mSet)}/{nSensiblePoints}\t\t{len(minPointset)}/{nSensiblePoints}\t\t\t{avgNBetterMinimums:.2f}/{nDistinctValues}')


