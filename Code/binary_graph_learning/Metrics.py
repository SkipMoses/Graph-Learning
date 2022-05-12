import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy import sparse
from pygsp import graphs, filters
from numpy.linalg import inv
import itertools as it

# Count the edges given a threshhold
def count_edges(Lap, threshhold):
    count = 0
    size = Lap.shape[0]
    for i in range(0,size):
        for j in range(0,i):
            if abs(Lap[i][j]) > threshhold:
                count = count + 1
    if count > 0:
        return(count)
    else:
        return(-1)

def detect_edges(L1, L2, threshhold):
    count = 0
    size = L1.shape[0]
    for i in range(0,size):
        for j in range(0,i):
            if (abs(L1[i][j]) > threshhold) and L2[i][j] > 0:
                count = count + 1
    return(count)

# Count the number of edges two laplacians have in common. 
# Computes the difference and counts the number of components
# sufficiently close to 0.
def compare_edges_strict(L1, L2):
    Dif = abs(L1 - L2)
    size = Dif.shape[0]
    count = 0
    for i in range(0,size):
        for j in range(0,i):
            if Dif[i][j] < .005:
                count = count + 1
    return(count)

# Computes the proportion of correct edges in the Learned Laplacian
def Precision(EstL, GTL, threshhold):
    return(detect_edges(EstL, GTL, threshhold)/count_edges(EstL, threshhold))

# Computes the proportion edges in Ground truth Laplacian 
# that appear in learned Laplacian
def Recall(EstL, GTL, threshhold):
    return(detect_edges(EstL, GTL, threshhold)/count_edges(GTL, 0))

def F_Measure(EstL, GTL, threshhold):
    P = Precision(EstL, GTL, threshhold)
    R = Recall(EstL, GTL, threshhold)
    if P+R > 0:
        return(2*((P*R)/(P+R)))
    else:
        return(-1)

def Relative_Error(L1, L2):
    return(np.linalg.norm(L1 - L2)/np.linalg.norm(L1))

def SSE(L1, L2):
    return(Relative_Error(L1,L2)**2)

def ComputeMetrics(EstL, GTL, threshhold):
    return({"Precision":Precision(EstL, GTL, threshhold),
          "Recall":Recall(EstL, GTL, threshhold),
          "F-Measure":F_Measure(EstL, GTL, threshhold),
          "Relative Error":Relative_Error(EstL, GTL),
          "SSE":SSE(EstL, GTL)})