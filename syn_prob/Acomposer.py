import torch
import random

def composeA(m, n, minPoint, b, visProb, onePer):
    """
    Composes a binary matrix `A` of size (m, n) such that minPoint is a feasible point and at least onePer of the entries are set to one.
    
    Args:
        m (int): The number of rows in the matrix `A`.
        n (int): The number of columns in the matrix `A`.
        minPoint (list): A binary vector of size `n` representing the minimum point, where 1s indicate selected columns.
        b (int): The number of guaranteed 1s in each row, randomly selected from the columns indicated by 1 in `minPoint`.
        visProb (float): Probability of adding additional 1s in the columns indicated by `minPoint` but not selected initially.
        onePer (float): The desired proportion of 1s in the entire matrix `A`.

    Returns:
        A (torch.Tensor): A binary matrix of size (m, n), where each row has exactly `b` guaranteed 1s and additional 1s are 
                          placed with probability `visProb`. The total number of 1s in the matrix is approximately `m * n * onePer`.
    """


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