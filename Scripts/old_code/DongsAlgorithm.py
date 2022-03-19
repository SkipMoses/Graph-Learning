import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy import sparse
from pygsp import graphs, filters
from numpy.linalg import inv
import itertools as it

def Problem1(a, b, Signal):
    # Number of rows in our Data set.
    num_vertices = Signal.shape[0]
    
    # Constant vectors of 1's and 0's
    ONE = np.ones(num_vertices)
    ZERO = np.zeros(num_vertices)
    
    # Variable for our objective function
    Laplacian = cp.Variable((num_vertices, num_vertices), symmetric = True)
    
    # Constaints for valid Laplacian
    constraints = [
                   cp.trace(Laplacian) == num_vertices,                       
                   Laplacian - cp.diag(cp.diag(Laplacian)) <= 0,
                   cp.diag(Laplacian) >= 0,
                   Laplacian@ONE == 0
                  ]
    
    # Objective Function to be minimized
    obj = cp.Minimize(a*cp.trace(Signal.transpose()@Laplacian@Signal) + 
                      b*cp.norm(Laplacian)**2)
    
    prob = cp.Problem(obj, constraints)
    prob.solve()
    return np.matrix(Laplacian.value)

def GL_SigRep(a, b, Signal, tol, max_itr):
    Y = Signal
    I = np.matrix(np.identity(Signal.shape[0]))
    L = Problem1(a, b, Y)
    for i in range(1,max_itr):
        temp = L
        
        # Closed form solution given by Dong et al.
        Y = (inv(I + a*L))@(Signal)
        
        # Find a new L given Y.
        L = Problem1(a, b, Y)
        
        # Check if we are with in the tolerance
        if(np.all(abs(temp - L) < tol)):
                return np.asarray(L)
    return np.asarray(L)