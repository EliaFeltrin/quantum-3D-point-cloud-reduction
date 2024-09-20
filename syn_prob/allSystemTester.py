import torch
import math
import trainer as tr
import solver as sl
import Acomposer as cm

def updateAvg(avg, iter, newVal):
    if(iter == 0):
        return newVal
    else:
        return (avg * iter + newVal) / (iter + 1)
    
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

epochs = 1
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
check = True

print("iter\t\t|OK\t\t|avgMinVal\t|avgVal\t\t|avgMaxVal\t|avgValm\t|avg #ones @min\t\t|# distinct m\t\t|# distinct min point\t|avg # bett. min\t|best epoch\t|avg nZerosUpperTriangQ\t|avg Qmean\t|avg QstdDev")

for iter in range(0, nTest):
    wellDone, global_min_value, global_max_value, avgValue, valueAtm, nOnesIn_m, nOnesInActualMin, m, minPoint, nSensiblePoints, nBetterMinimums, nDistinctValues, bestQEpoch, nZerosUpperTriangBestQ, bestQmean, bestQstd, F, Q, m = tr.trainOne(nSelectedPointAtMinimum, n, tFs, tsPerc, MVal, mVal, overlap, batchSize, Q_init_type, verbose, mse_loss_adj_factor, constraint_loss_adj_factor, min_loss_adj_factor, mean_std_loss_adj_factor, randomMeanInit, randomStdInit, printEpochs, epochs, forcedMinValQUT, printFinalQs, check, b)
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

    A = cm.composeA(nImages, n, m, b, visProb, onePer)

    Q = Q[0]
    print(type(Q))
    print(Q)

    print(type(A))
    print(A)

    x_optimal, objective_value = sl.solve_bqp(Q, A, b)

    print(f"m:\t{m}:{valueAtm}\nopt:\t{x_optimal}:{objective_value}")


    # for f in F:
    #     oneCounterMap[f] = f.count(1)

    # for k, v in sorted(oneCounterMap.items(), key=lambda x: x[1]):
    #     print(f'{k}: {v}')

    

