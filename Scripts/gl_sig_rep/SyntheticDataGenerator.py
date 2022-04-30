import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import networkx as nx
from scipy import sparse
from numpy.linalg import inv
import itertools as it

# Generate an num_signalsXnum_Vertices matrix of synthetic data
# over a graph G following Dong's model. 
# mu := mean; sigma := standard deviation
def RandomSignal(graph, num_signals, mu, sigma, seed):
    # The graph Laplacian
    L = nx.laplacian_matrix(graph).toarray()
    
    # The number of vertices
    size = L.shape[0]
    
    
    # Normalized Laplacian ie tr(L) = num_vertices
    NormL = (size/np.trace(L))*L
    
    # Moore Penrose Psudoinverse
    MPPI = np.linalg.pinv(NormL, hermitian= True)
    
    # Vector of means
    mean = mu*np.ones(size)
    
    # Matrix of standard deviation
    stan_dev = MPPI + sigma*np.diag(np.ones(size))
    X = []
    for i in range(num_signals):
        np.random.seed(seed + i)
        signal = np.random.multivariate_normal(mean, stan_dev)
        X.append(signal.tolist())
    return np.transpose(X)
    



# Generates Synthetic data following Dongs Model
# with an added regression term
# Generate an num_signalsXnum_Vertices matrix of synthetic data
# over a graph G following Dong's model. 
# mu := mean; sigma := standard deviation
def RandomRegressorSignal(graph, mu, sigma, b, P, seed):
    # The Graph Laplacian
    L = nx.laplacian_matrix(graph).toarray()
    
    # The number of vertices
    size = L.shape[0]
    
    if(size != P[0].shape[0]):
        print("Number vertices does not equal number of rows in P.")
        return
    if(P[0].shape[1] != b.shape[0]):
        print("Number of columns of P[i] is not equal to number of rows of b.")
        return
    
    # Normalized Laplacian ie tr(L) = num_vertices
    NormL = (size/np.trace(L))*L
    
    # Moore Penrose Psudoinverse
    MPPI = np.linalg.pinv(NormL, hermitian= True)
    
    # Vector of means
    mean = mu*np.ones(size)
    
    # Matrix of standard deviation
    stan_dev = MPPI + sigma*np.diag(np.ones(size))
    X = []
    np.random.seed(seed)
        
    # Create random regression coefficients 
    #b = np.random.rand(observations,)
    #b = np.full(shape=observations, fill_value=1, dtype=np.int)
    for i in range(len(P)):
        np.random.seed(seed + i)
        signal = np.random.multivariate_normal(mean, stan_dev) + P[i]@b
        X.append(signal.tolist())
    return np.matrix(np.transpose(X))

# Generates synthetic data of a specific distribution
# num_observations: Length of the column
# distribution:     (string) Type of distribution 
# parameters:       (string) Parameters for the distribution 
# Example:          generate_data_type(1,1,'uniform', '0,1')
def generate_data_type(num_observation, distribution, parameters, seed):
    rand = np.random.RandomState(seed)
    data_table = []
    x = eval('rand.'+distribution+'('+parameters+', '+str(num_observation)+')')
    x = [i for i in x]
    data_table.append(x)
    return(np.transpose(np.matrix(data_table)))

# Uses generate_data_type to create synthetic data for numerous 
# distributions
# num_observations: (Int) Length of a row in the table
# predictors_vec:   (list<int>) Number of predictors of a given distribution
# type.
# distributions:    (list<str>) List of the distribution types in the data
# parameters:       (list<list><str>) List of parameters for corresponding distirbution
def generate_experiment(num_observations, distributions, parameters, seed):
    s = seed
    M_shape = (num_observations, sum([len(i) for i in parameters]))
    Ptemp = [i for i in parameters]
    M = generate_data_type(num_observations, distributions[0], Ptemp[0][0], s)
    Ptemp[0] = Ptemp[0][1:len(Ptemp)]
    for i in range(len(Ptemp)):
        for p in Ptemp[i]:
            s = s + 1
            temp = generate_data_type(num_observations, distributions[i], p, s)
            M = np.concatenate((M,temp))
    M = np.reshape(M, M_shape, 'F')
    return(M)
def generate_trials(num_observations, num_trials, distributions, parameters, seed):
    s = seed
    T = []
    for i in range(num_trials):
        P = generate_experiment(num_observations, distributions, parameters, s)
        T.append(P)
    return(T)