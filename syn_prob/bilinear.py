import torch
import random
import torch.nn as nn
import math

def generate_random_binary_vector(n, n_ones):
    """
    Function to generate a random binary vetor of lenght n with n_ones ones

    :param n: vector length
    :type n: int
    :param n_ones: number of ones
    :type n_ones: int
    :return: vector
    :rtype: torch tensor
    """
    # Initialize a zero vector of length n
    vector = torch.zeros(n, dtype=torch.float32)
    
    # Randomly choose `n_ones` unique indices to set to 1
    indices = torch.randperm(n)[:n_ones]
    
    # Set the chosen indices to 1
    vector[indices] = 1
    
    return vector

def generate_F_set(n, l, m_vector):
    """
    Function to generate a (sub-)set of feasible points needed to train the bilinear leayer.
    The point are selected at increasing hamming distance from the chosen minimum point
    Thee function computes the coefficent for a function hamming_distance -> n_of_point 
    ensuring that at least l points are createdand heach hamming_distance is covered

    :param n: Q size
    :type n: int
    :param l: desired length of the F set
    :type l: int
    :param m_vector: minimum point
    :type m_vector: torch tensor
    :return: F set
    :rtype: set
    """

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
    """
    Function to compute the hamming distance between two binary vectors.

    :param v1: 
    :type v1: torch tensor
    :param v2: 
    :type v2: torch tensor
    :return:
    :rtype: int
    """

    return (v1 != v2).sum().item()

def create_map(F, MVal, mVal, m, overlap_factor=0.1):
    """
    Creates a mapping between a set of binary vectors `F` and corresponding values, 
    based on their Hamming distance from a minimum point `m`.

    The function generates a dictionary `M` where each binary vector in `F` is 
    assigned a value between `mVal` and `MVal`. The value is determined based on 
    the Hamming distance from the vector `m`, such that vectors closer to `m` get 
    values closer to `mVal`, and vectors further away from `m` get values closer 
    to `MVal`. An overlap factor introduces randomness in the value assignment 
    by creating a range for the values based on the distance.

    Args:
        F (iterable): A collection of binary vectors (tuples) to be mapped.
        MVal (float): The maximum value to assign to a vector in the map.
        mVal (float): The minimum value to assign to the vector `m`, 
                      corresponding to a Hamming distance of 0.
        m (torch.Tensor): The reference binary vector used to compute Hamming distances.
        overlap_factor (float, optional): A factor controlling the amount of overlap 
                                          between the ranges of values for different 
                                          distances. Defaults to 0.1.

    Returns:
        dict: A dictionary mapping each binary vector in `F` to a value based on 
              its Hamming distance from `m`, with some added randomness.
    """


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

class BilinearLayer(nn.Module):
    """
    A custom bilinear layer that supports various initialization strategies for the 
    weight tensor. This layer applies a bilinear transformation between two input 
    vectors and outputs a scalar value, with the weight matrix being constrained 
    to be upper triangular with diagonal and being >= forcedMinValQUT during both forward and backward passes.

    Args:
        n (int): The size of the input vectors (must be the same for both inputs).
        init_type (str, optional): The type of initialization for the bilinear weight 
                                   matrix. Options include:
            - 'id': Initializes the weight as an identity matrix.
            - 'near_id': Initializes the weight as a matrix close to identity with 
                         small random noise.
            - 'psd': Initializes the weight to be positive semi-definite.
            - 'm_biased': Initializes the weight matrix biased towards a vector `m` 
                          by computing its outer product.
            - 'random': Random initialization with additional constraints.
        m (torch.Tensor, optional): A reference vector used for the 'm_biased' 
                                    initialization. Required if `init_type='m_biased'`.

    Methods:
        initialize_weights(init_type, m, verbose, randomMeanInit, randomStdInit, forcedMinValQUT):
            Initializes the weight tensor of the bilinear layer based on the selected
            strategy. Ensures the weight matrix is initialized according to specific 
            properties (e.g., identity, positive semi-definiteness, or random 
            distribution). It enforces that the upper triangular part of the matrix is 
            >= forcedMinValQUT and optionally prints the initialized weight matrix if `verbose` 
            is True.
        
        forward(x1, x2, forcedMinValQUT):
            Performs the forward pass, enforcing the weight matrix to be upper triangular 
            and positive. The `forcedMinValQUT` ensures that all values are constrained 
            to be greater than or equal to the specified threshold in the upper triangular 
            part of the matrix, and it applies the bilinear transformation between inputs 
            `x1` and `x2`.
        
        enforce_upper_triangular_and_positive(grad, forcedMinValQUT):
            A hook function applied to the weight tensor after each backward pass. It 
            ensures that the weight matrix remains upper triangular and positive, with 
            all values in the upper triangular part being constrained to be greater than 
            or equal to `forcedMinValQUT`.

    Raises:
        ValueError: If the input dimensions do not match for certain initialization 
                    types (e.g., identity, near_id), or if `m` is not provided for 
                    'm_biased' initialization.
    """


    def __init__(self, n, verbose, forcedMinValQUT, init_type='identity', m=None, randomMeanInit=None, randomStdInit=None):
        super(BilinearLayer, self).__init__()
        self.bilinear = nn.Bilinear(n, n, 1, bias=False)  # No bias
        self.forcedMinValQUT = forcedMinValQUT

        # Initialize the weight based on the selected strategy
        self.initialize_weights(init_type, m, verbose, randomMeanInit, randomStdInit)

    def initialize_weights(self, init_type, m, verbose, randomMeanInit, randomStdInit):
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
            if randomMeanInit is None or randomStdInit is None:
                raise ValueError(f"one or more params missing")
            # Random initialization (PyTorch default)
            nn.init.normal_(weight, mean=randomMeanInit, std=randomStdInit)
            with torch.no_grad():
                Q = self.bilinear.weight
                Q.clamp_(min=self.forcedMinValQUT)  # Ensure all values are >= than forcedMinValQUT in the upper triang part
                Q.copy_(torch.triu(Q))  # Ensure upper triangular

        else:
            raise ValueError(f"Unknown initialization type: {init_type}")
        
        if(verbose):
                print("Q init:")
                print(Q)
                print("\n")

        # Register backward hook to modify the weights after each backward pass
        self.bilinear.weight.register_hook(self.enforce_upper_triangular_and_positive)
    
    def forward(self, x1, x2):
        # Force the weight matrix to be upper triangular and positive in the forward pass
        with torch.no_grad():
            Q = self.bilinear.weight
            Q.clamp_(min=self.forcedMinValQUT)  # Ensure all values are >= than forcedMinValQUT in the upper triang part
            Q.copy_(torch.triu(Q))  # Ensure upper triangular

        return self.bilinear(x1, x2)

    def enforce_upper_triangular_and_positive(self, grad):
        """ Hook function that forces the weight to be upper triangular and positive after each backward pass. """
        with torch.no_grad():
            Q = self.bilinear.weight
            Q.clamp_(min=self.forcedMinValQUT)  # Ensure all values are >= than forcedMinValQUT in the upper triang part
            Q.copy_(torch.triu(Q))  # Ensure the matrix is upper triangular
        
        return grad  # Return the gradient unmodified (for normal backward pass)
    
def custom_loss(Q, n):
    """
    Computes a custom loss by counting the number of zero elements in the upper triangular 
    part of a bilinear weight matrix `Q`. The loss is simply the count of zeros in the 
    upper triangular portion of the matrix.

    Args:
        Q (torch.Tensor): A 3D tensor representing the bilinear weight matrix, where the 
                          first dimension is the batch size and the second and third 
                          dimensions represent the matrix to be evaluated.
        n (int): The size of the square matrix (number of rows/columns).

    Returns:
        torch.Tensor: A tensor containing the count of zero elements in the upper triangular 
                      part of the matrix `Q`.
    """

    zeroCounter = 0
    for i in range(0,n):
        for j in range(i,n):
            if Q[0][i][j].item() == 0:
                zeroCounter += 1

    return torch.tensor(zeroCounter)

def global_minimum_loss(Q, m_tensor):
    """
    Computes the global minimum loss by evaluating the quadratic form of the matrix `Q`
    at the vector `m`. This loss encourages the matrix `Q` to have a low value
    at the point `m`.

    Args:
        Q (torch.Tensor): A 2D tensor (matrix) that represents the bilinear weight matrix.
        m_tensor (torch.Tensor): A 1D tensor representing the vector `m`, where the 
                                 minimum value is desired.

    Returns:
        torch.Tensor: The value of the quadratic form `m^T * Q * m`, which represents the 
                      minimum value of the matrix `Q` at `m`. This value can be used as a 
                      loss to penalize `Q` if the value at `m` is not sufficiently small
    """

    # Compute m^T * Q * m
    m_tensor = m_tensor.unsqueeze(0)
    min_value_at_m = torch.matmul(m_tensor, torch.matmul(Q, m_tensor.T))
    
    # Penalize if the value at m is not sufficiently small (encourage Q to minimize at m)
    #return torch.relu(min_value_at_m)
    return min_value_at_m
    #return torch.sigmoid(min_value_at_m)   #the smaller the best, even negative

def avgStdLoss(Q, m1, sigma1):    
    """
    Computes a loss based on the comparison between the mean (`m1`) and standard deviation 
    (`sigma1`) of a reference distribution, and the mean (`m2`) and standard deviation (`sigma2`) 
    of the upper triangular elements of the matrix `Q`. The loss is derived from a statistical 
    similarity measure between the two distributions.

    Args:
        Q (torch.Tensor): A 3D tensor where the first dimension is the batch size and the second 
                          and third dimensions represent a square matrix. The loss is computed 
                          using the upper triangular elements of this matrix.
        m1 (float): The mean of the reference distribution.
        sigma1 (float): The standard deviation of the reference distribution.

    Returns:
        torch.Tensor: A scalar tensor representing the loss, which is based on the statistical 
                      similarity between the upper triangular elements of `Q` and the reference 
                      distribution characterized by `m1` and `sigma1`.
    """

    n = Q.size()[1]

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

def makeQUpperTriangular(Q, forcedMinValQUT):
    """
    Converts a given matrix `Q` into an upper triangular matrix by setting all elements 
    below the main diagonal to zero. Additionally, it ensures that all elements in the 
    upper triangular part of the matrix are at least equal to a specified minimum value 
    (`forcedMinValQUT`).

    Args:
        Q (torch.Tensor): A 2D tensor (matrix) that will be modified to become upper 
                          triangular. It is expected to be a square matrix.
        forcedMinValQUT (float): the minimum value allowed in the upper-triangular and diagonal part

    Returns:
        None: The function modifies the input matrix `Q` in-place to ensure it is 
              upper triangular and adheres to the minimum value constraint.
    """

    n = Q.size()[1]

    for i in range(0,n):
        for j in range(0, n):
            if(j<i):            #upper triangular
                Q[i,j] = 0
            else:
                Q[i,j] = max(Q[i,j], forcedMinValQUT) 