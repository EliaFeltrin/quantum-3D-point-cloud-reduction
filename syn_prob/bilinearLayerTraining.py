import torch
import torch.nn as nn
import torch.optim as optim
import random
import itertools
import math
import statistics
import gurobipy as gp
from gurobipy import GRB
import numpy as np

def generate_random_binary_vector(n, n_ones):
    # Initialize a zero vector of length n
    vector = torch.zeros(n, dtype=torch.float32)
    
    # Randomly choose `n_ones` unique indices to set to 1
    indices = torch.randperm(n)[:n_ones]
    
    # Set the chosen indices to 1
    vector[indices] = 1
    
    return vector

def generate_F_set(n, l, m_vector):
    # Compute the coefficient A of the hyperbolic function that determines the number of points in the training set based on increasing distance from m
    denom = sum(1 / i for i in range(1, n + 1))
    A = l / denom

    # Initialize set F and add the original vector
    F = set()
    F.add(tuple(m_vector.tolist()))  # Add the initial m_vector to the set
    nPoint = 0
    j = 1

    # Populating F with at least l points
    while len(F) < l:
        oldFlen = len(F)
        part = math.ceil(A / j)

        # Generating new vectors with `j` bits of distance from `m_vector`
        while len(F) - oldFlen < part:
            # Create a copy of m_vector
            vector = m_vector.clone()
            # Generate `j` random positions to flip
            flip_positions = random.sample(range(n), j)
            vector[flip_positions] = 1 - vector[flip_positions]  # Flip the bits at selected positions

            # Add the new vector to F
            F.add(tuple(vector.tolist()))

        nPoint += part
        j += 1

    return F, len(F)
    

def hamming_distance(v1, v2):
    # Calculate the Hamming distance between two binary vectors
    return (v1 != v2).sum().item()

def create_map(F, MVal, mVal, m, overlap_factor=0.1):
    # Initialize the map M
    M = {}
    
    # Compute the maximum possible Hamming distance
    max_dist = len(m)
    
    # Assign values based on the Hamming distance
    for f_tuple in F:
        f = torch.tensor(f_tuple)
        dist = hamming_distance(f, m)
        if dist == 0:
            M[f_tuple] = mVal  # Map m to the lowest value
        else:
            # The value should increase as the distance increases, but with some overlap
            range_min = mVal + (dist / max_dist) * (MVal - mVal)
            range_max = range_min + overlap_factor * (MVal - mVal)
            M[f_tuple] = random.uniform(range_min, min(range_max, MVal))
    
    return M

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
            nn.init.normal_(weight, mean=randomMeanInit, std=randomStdInit)
            with torch.no_grad():
                Q = self.bilinear.weight
                Q.clamp_(min=forcedMinValQUT)  # Ensure all values are positive
                Q.copy_(torch.triu(Q))  # Ensure upper triangular
            if(verbose):
                print("Q init:")
                print(Q)
                print("\n")

        else:
            raise ValueError(f"Unknown initialization type: {init_type}")

        # Register backward hook to modify the weights after each backward pass
        self.bilinear.weight.register_hook(self.enforce_upper_triangular_and_positive)

    def forward(self, x1, x2):
        # Force the weight matrix to be upper triangular and positive in the forward pass
        with torch.no_grad():
            Q = self.bilinear.weight
            Q.clamp_(min=forcedMinValQUT)  # Ensure all values are positive
            Q.copy_(torch.triu(Q))  # Ensure upper triangular

        return self.bilinear(x1, x2)

    def enforce_upper_triangular_and_positive(self, grad):
        """ Hook function that forces the weight to be upper triangular and positive after each backward pass. """
        with torch.no_grad():
            Q = self.bilinear.weight
            Q.clamp_(min=forcedMinValQUT)  # Ensure all values remain positive
            Q.copy_(torch.triu(Q))  # Ensure the matrix is upper triangular
        
        return grad  # Return the gradient unmodified (for normal backward pass)



def custom_loss(Q, n):
    #print(Q)
    zeroCounter = 0
    for i in range(0,n):
        for j in range(i,n):
            if Q[0][i][j].item() == 0:
                zeroCounter += 1

    #print(zeroCounter)

    return torch.tensor(zeroCounter)

def global_minimum_loss(Q, m_tensor):
    # Compute m^T * Q * m
    m_tensor = m_tensor.unsqueeze(0)
    min_value_at_m = torch.matmul(m_tensor, torch.matmul(Q, m_tensor.T))
    
    # Penalize if the value at m is not sufficiently small (encourage Q to minimize at m)
    #return torch.relu(min_value_at_m)
    return min_value_at_m
    #return torch.sigmoid(min_value_at_m)   #the smaller the best, even negative

def avgStdLoss(Q, m1, sigma1):    
    triuQ = []
    for i in range(0, n):
        for j in range(i, n):
            triuQ.append(Q[0][i][j])

    triuQ = torch.tensor(triuQ)
    
    m2 = torch.mean(triuQ)
    sigma2 = torch.std(triuQ)

    coeff = torch.sqrt(2 * sigma1 * sigma2 / (sigma1**2 + sigma2**2))
    exp_term = torch.exp(- (m1 - m2)**2 / (4 * (sigma1**2 + sigma2**2)))
    
    return coeff * exp_term

# Function to generate all possible binary vectors of length n but 0
def generate_all_binary_vectors(n, b):
    return [torch.tensor(list(vec), dtype=torch.float32) for vec in itertools.product([0, 1], repeat=n) if vec.count(1) >= b]

def makeQUpperTriangular(Q):
    for i in range(0,n):
        for j in range(0, n):
            if(j<i):            #upper triangular
                Q[i,j] = 0
            else:
                Q[i,j] = max(Q[i,j], forcedMinValQUT)   #positive values

def makeQPositiveValued(Q):
    for i in range(0,n):
        for j in range(0,n):
            Q[i,j] = abs(Q[i,j])


def trainOne():
    # Generate set F and map M
    m = generate_random_binary_vector(n, nSelectedPointAtMinimum)
    F, aFs  = generate_F_set(n, tFs, m)
    tsSize  = int(aFs * tsPerc)
    map = create_map(F, MVal, mVal, m, overlap)

    batchSize_ = min(batchSize, aFs)

    if(verbose):
        print("Problem dimensionality: ", n)
        print("Size of the training set: ", aFs)
        print("Size of the test set: ", tsSize)
        print(f"Selected minimum point m (value {mVal:.1f}): {[int(e) for e in m.tolist()]}\t ones at indexes {[i+1 for i, x in enumerate(m.tolist()) if x == 1]}")
        print("Training set:")
        for k, v in sorted(map.items(), key=lambda x: x[1]):
            print(f'{list(k)}, {v:.2f}\thamming distance: {hamming_distance(torch.tensor(k), m)}')
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
        total_mean_std_loss = 0
        Q = None
        random.shuffle(train_pairs)  # Shuffle training pairs for each epoch
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
            constraint_loss = custom_loss(Q, n) * constraint_loss_adj_factor
            min_loss = global_minimum_loss(Q, m.clone().detach().requires_grad_(True)) * min_loss_adj_factor
            mean_std_loss = avgStdLoss(Q, randomMeanInit, randomStdInit) * mean_std_loss_adj_factor

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
        ok = currentEpoch < epochs #and abs(oldLoss - total_loss) > 0.1
        if not ok and (currentEpoch + 1) % printEpochs != 0 and verbose:
            print(f'Epoch {currentEpoch+1}/{epochs}, Loss: {total_mse_loss:.1f}\t{total_constraint_loss:.1f}\t{total_min_loss:.1f} + {total_mean_std_loss:.4f} = {total_loss:.4f}')
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
    if(verbose):
        print("Best learned matrix Q at epoch ", best_Q_epoch)
        print(Q)

    Q = Q.detach()

    makeQUpperTriangular(Q)
    #makeQPositiveValued(Q)

    if(printFinalQs):
        print("\nZeroed Q:")
        print(Q)

        print("\n############################################################################################################\n")


    # Compute m^T * Q * m (the target minimum value)
    min_value_at_m = torch.matmul(m, torch.matmul(Q, m)).item()
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
    res = {}                             
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

    if(verbose):
        print("\n############################################################################################################\n")

        
    # Print the value at each vector
    if(verbose):
        counter = 0
        for k, v in sorted(res.items(), key=lambda x: x[1]):
            if counter >= 50:
                break
            print(f"Value at x = {[int(xi) for xi in list(k)]}: {v:.2f}\t#hamming = {hamming_distance(torch.tensor(k), m)}\t#selected points = {k.count(1   )}")
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

    nZerosUpperTriangBestQ = custom_loss(best_Q, n)

    triuQ = []
    for i in range(0, n):
        for j in range(i, n):
            triuQ.append(Q[i][j].item())


    bestQmean = statistics.mean(triuQ)
    bestQstd = statistics.stdev(triuQ)

    return wellDone, global_min_value, global_max_value, avgValue, valueAtm, m.tolist().count(1), min_point.tolist().count(1), m, tuple(min_point.tolist()), len(all_vectors), nBetterMinimums, len(values), best_Q_epoch, nZerosUpperTriangBestQ, bestQmean, bestQstd, F, best_Q, m

def updateAvg(avg, iter, newVal):
    if(iter == 0):
        return newVal
    else:
        return (avg * iter + newVal) / (iter + 1)
    
def composeA(m, n, minPoint, b, visProb, onePer):
    A = torch.zeros(m, n, dtype = torch.float32)
    oneIndexes = [i for i, x in enumerate(list(minPoint)) if x == 1]

    freePos = list(range(0, m*n))

    for r in range(0,m):
        missing = oneIndexes.copy()
        for c in random.sample(oneIndexes, b):
            A[r][c] = 1
            missing.remove(c)
            freePos.remove(r*n + c)

        for c in missing:
            if random.random() < visProb:
                A[r][c] = 1
                freePos.remove(r*n + c)

    nOnes = (A == 1).sum().item()
    for compInd in random.sample(freePos, int(m*n*onePer - nOnes)):
        A[compInd // n][compInd % n] = 1


    return A




# Function to solve the binary quadratic programming problem
def solve_bqp(Q, A, b, m):
    # Convert torch tensors Q and A to numpy arrays
    Q_np = Q.detach().cpu().numpy()  # Ensure it's on CPU and then convert to numpy
    A_np = A.detach().cpu().numpy()
    
    b_np = np.array([b for _ in range(0, m)])

    # Number of variables (length of x)
    n = Q_np.shape[0]
    
    # Create a new model
    model = gp.Model("binary_quadratic_program")
    
    # Create binary variables (x_i in {0, 1})
    x = model.addVars(n, vtype=GRB.BINARY, name="x")
    
    # Set the objective function: min x^T Q x
    # Since x^T Q x is a quadratic expression, we use the 'QuadExpr' in Gurobi
    obj = gp.QuadExpr()
    for i in range(n):
        for j in range(i, n):  # Upper triangular part only
            obj += Q_np[i, j] * x[i] * x[j]
    
    model.setObjective(obj, GRB.MINIMIZE)
    
    # Add constraints: Ax >= b
    for i in range(A_np.shape[0]):  # Iterate over each constraint
        expr = gp.LinExpr()
        for j in range(n):
            expr += A_np[i, j] * x[j]
        model.addConstr(expr >= b_np[i], name=f"constraint_{i}")
    
    # Optimize the model
    model.optimize()
    
    # Extract the optimal solution
    if model.status == GRB.OPTIMAL:
        x_opt = np.array([x[i].x for i in range(n)])
        print(f"Optimal solution: {x_opt}")
        print(f"Objective value: {model.ObjVal}")
        return x_opt, model.ObjVal
    else:
        print("No optimal solution found.")
        return None, None


# Parameters

torch.set_printoptions(linewidth=500)

nSelectedPointAtMinimum = 3
b = 2                       # minimum number of point that amust be visible for each image
n = 20                      # problem dimensionality
tFs = 50
tFs = min(tFs, math.comb(n, (n-nSelectedPointAtMinimum)))     # tentative size of the subset of feasible point (generated as random, the duplicates are removed)
tsPerc = 0.5                # size of the test set
MVal = 100.0                 # maximum value possibly reachable by the func
mVal = -100.0
overlap = 0.1
forcedMinValQUT = float('-inf')
randomMeanInit = 0
randomStdInit = 0.2

epochs = 150
printEpochs = 5
batchSize = 32
mse_loss_adj_factor = 1.0
constraint_loss_adj_factor = 1.0    #forcing the matrix in the  forward/backward pass will result in this loss being 0 
min_loss_adj_factor = 1.0
mean_std_loss_adj_factor = 100.0    
Q_init_type = 'random'          #choose between id, near_id, psd, m_biased, random
nTest = 1

nImages = 6

verbose = True
visProb = 0.2
onePer = 0.3

totalOk = 0
avgMinVal = 0
avgMaxVal = 0
avgAvgVal = 0
avgValueAtm = 0
avgNOnesIn_m = 0
avgNOnesInActualMin = 0 
avgNBetterMinimums = 0
avgBestEpoch = 0
avgNZeroUTQ = 0
avgQmean = 0
avgQstd = 0
mSet = set([])
minPointset = set([])
if(nTest > 1):
    verbose = False
printFinalQs = False or verbose

print("iter\t\t|OK\t\t|avgMinVal\t|avgVal\t\t|avgMaxVal\t|avgValm\t|avg #ones @min\t\t|# distinct m\t\t|# distinct min point\t|avg # bett. min\t|best epoch\t|avg nZerosUpperTriangQ\t|avg Qmean\t|avg QstdDev")

for iter in range(0, nTest):
    wellDone, global_min_value, global_max_value, avgValue, valueAtm, nOnesIn_m, nOnesInActualMin, m, minPoint, nSensiblePoints, nBetterMinimums, nDistinctValues, bestQEpoch, nZerosUpperTriangBestQ, bestQmean, bestQstd, F, Q, m = trainOne()
    totalOk += wellDone
    avgMinVal = updateAvg(avgMinVal, iter, global_min_value)
    avgAvgVal = updateAvg(avgAvgVal, iter, avgValue)
    avgMaxVal = updateAvg(avgMaxVal, iter, global_max_value)
    avgValueAtm = updateAvg(avgValueAtm, iter, valueAtm)
    avgNOnesIn_m = updateAvg(avgNOnesIn_m, iter, nOnesIn_m)
    avgNOnesInActualMin = updateAvg(avgNOnesInActualMin, iter, nOnesInActualMin)
    #if(not wellDone):
    avgNBetterMinimums = updateAvg(avgNBetterMinimums, iter, nBetterMinimums)
    mSet.add(m)
    minPointset.add(minPoint)
    avgBestEpoch = updateAvg(avgBestEpoch, iter, bestQEpoch)
    avgNZeroUTQ = updateAvg(avgNZeroUTQ, iter, nZerosUpperTriangBestQ)
    avgQmean = updateAvg(avgQmean, iter, bestQmean)
    avgQstd = updateAvg(avgQstd, iter, bestQstd)

    oneCounterMap = {}
    print(f'{iter}\t\t{totalOk}/{iter+1}\t\t{avgMinVal:.2f}\t\t{avgAvgVal:.2f}\t\t{avgMaxVal:.2f}\t\t{avgValueAtm:.2f}\t\t{avgNOnesInActualMin:.2f}/{n}\t\t{len(mSet)}/{nSensiblePoints}\t\t\t{len(minPointset)}/{nSensiblePoints}\t\t\t{avgNBetterMinimums:.2f}/{nDistinctValues}\t\t{avgBestEpoch:.2f}\t\t{avgNZeroUTQ:.2f}\t\t\t{avgQmean:.2f}\t\t{avgQstd:.2f}')

    A = composeA(nImages, n, m, b, visProb, onePer)

    Q = Q[0]
    print(type(Q))
    print(Q)

    print(type(A))
    print(A)

    x_optimal, objective_value = solve_bqp(Q, A, b, nImages)

    print(f"m:\t{m}:{valueAtm}\nopt:\t{x_optimal}:{objective_value}")


    # for f in F:
    #     oneCounterMap[f] = f.count(1)

    # for k, v in sorted(oneCounterMap.items(), key=lambda x: x[1]):
    #     print(f'{k}: {v}')

    

