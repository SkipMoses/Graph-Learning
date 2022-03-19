import numpy as np
import numpy as np
import scipy.sparse as sps
import itril as it

"""
function M = DuplicationM(n, option)
% function M = DuplicationM(n)
% M = DuplicationM(n, 'lo') % (default) OR
% M = DuplicationM(n, 'up') % 
% Return duplication matrix order n
%
% It is always assumed Duplication arount main diagonal (k=0)
%
% Ouput are sparse
%
% DuplicationM(size(A),'lo')*A(itril(size(A))) == A(:) %true for lower half
% DuplicationM(size(A),'up')*A(itriu(size(A))) == A(:) %true for upper half
%
% Author: Bruno Luong <brunoluong@yahoo.com>
% Date: 21/March/2009
%
% Ref : Magnus, Jan R.; Neudecker, Heinz (1980), "The elimination matrix:
% some lemmas and applications", Society for Industrial and Applied Mathematics.
% Journal on Algebraic and Discrete Methods 1 (4): 422449,  
% doi:10.1137/0601049, ISSN 0196-5212.
"""

def DuplicationM(n, option = 'lo'):
    if np.isscalar(n):
        n = [n, n]
    
    if option[0].lower() == 'l':
        I, J = it.itril(n,0,False)
    elif option[0].lower() == 'u':
        J, I = it.itril(n,0,False)
    else:
        print("Error, optioin mus be 'lo' or 'up'.")
        return()
    
    I = [x for _, x in sorted(zip(J, I))]
    J = sorted(J)
    
    # Find the sub/sup diagonal part that can flip to other side
    loctri = [i for i in range(len(I)) if 
              (I[i] != J[i]) and 
              (J[i] <= n[0]-1) and 
              (I[i] <= n[1]-1)]
    
    # Indices of the flipped part
    arg1 = [J[i] for i in loctri]
    arg2 = [I[i] for i in loctri]
    
    Itransposed = np.ravel_multi_index([arg1, arg2], n)
    
    # Convert to linear indice
    I = np.ravel_multi_index([I, J], n)
    
    arg1 = np.append(I,Itransposed)
    arg2 = np.append([i for i in range(len(I))], loctri)
    d = [1]*len(arg1)
    
    M = sps.csr_matrix((d, (arg1, arg2)), shape = (np.prod(n),len(I)))
    return(M)