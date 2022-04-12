import numpy as np
import optimize_laplacian_gaussian as olg

"""
function [L,Y,L_harvard] = graph_learning_gaussian(X_noisy,param)
% Learning graphs (Laplacian) from structured signals
% Signals X follow Gaussian assumption
"""
def graph_learning_gaussian(X_noisy, param):
    
    N = param['N']
    max_iter = param['max_iter']
    alpha = param['alpha']
    beta = param['beta']

    objective = [0]*max_iter
    Y_0 = X_noisy
    Y = Y_0
    
    for i in range(max_iter):
        
        # Step 1: given Y, update L
        L = olg.optimize_laplacian_gaussian(N,Y,alpha,beta)
        
        # Step 2: given L, update Y
        R = np.linalg.cholesky(np.identity(N) + alpha*L)
        Rt = np.transpose(R)
        arg1 = np.linalg.lstsq(Rt, Y_0)[0]
        # print('arg1 shape is ' + str(arg1.shape))
        # print('R shape is ' + str(R.shape))
        Y = np.linalg.lstsq(R, arg1)[0]
        
        # Store objective
        arg1 = np.linalg.norm(Y-Y_0, 'fro')**2 
        arg2 = alpha*(np.transpose((Y@np.transpose(Y)).flatten('F'))@(L.flatten('F')))
        arg3 = beta*np.linalg.norm(L, 'fro')**2
        objective[i] = arg1 + arg2 + arg3
        
        # Stopping criteria
        if i>=2 and abs(objective(i) - objective(i-1)) < 10**(-4):
            print(str(i) + ' iterations needed to converge.')
            break
        return([L.round(4), Y.round(4)])
