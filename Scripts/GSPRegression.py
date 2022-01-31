import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy import sparse
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
    prob.solve(solver = cp.SCS)
    return np.matrix(Laplacian.value)     

def GSP_Regression(a, b, Signal, P, tol, max_itr):
    Y = Signal
    R = np.zeros(Y.shape)
    I = np.matrix(np.identity(Signal.shape[0]))
    L = Problem1(a, b, Y)
    count = 0
    for i in range(1,max_itr):
        count = count + 1
        temp = L
        
        # Closed form solution given by Dong et al.
        Y = (inv(I + a*L))@(Signal - R)
        
        # Find new Regressor matrix
        R = Signal - Y
        
        # Find a new L given Y.
        L = Problem1(a, b, Y)
        
        # Check if we are within the tolerance
        if(np.all(abs(temp - L) < tol)):
                break
    #b = (inv(P.transpose()@P)@P.transpose)@np.matrix.flatten(R, 'F')
    #return [np.asarray(L), Y, b]
    avg_dif = np.mean(temp - L)
    print('Solution reached in ' + str(count) + ' iterations.')
    return [np.asarray(L), Y]