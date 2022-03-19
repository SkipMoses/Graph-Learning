import numpy as np

"""
# function [I J] = idiag(sz, k)
# function [I J] = idiag(sz, k) # OR
# I = itril(sz, k)
#
# Return the subindices [I J] (or linear indices I if single output call)
# in the purpose of extracting the diagonal of the matrix of the size SZ.
# Input k is optional shifting. For k=0, extract from the main
# diagonal. For k>0 -> above the diagonal, k<0 -> below the diagonal
#
# Output is a column and sorted with respect to linear indice
#
# Example:
#
# A = [ 7     5     4
#       4     2     3
#       9     1     9
#       3     5     7 ]
#
# I = idiag(size(A))  # gives [1 6 11]'
# A(I)                # gives [7 2 9]' OR diag(A)
#
# Author: Bruno Luong <brunoluong@yahoo.com>
# Date: 21/March/2009
"""
def idiag(sz, k = 0, linear_ind = True):
    if np.isscalar(sz):
        sz = [sz, sz]
    m = sz[0]
    n = sz[1]
    
    # Pay attention to clipping
    l = 0 - min(k,0)
    u = min(m,n-k)
    I = list(range(l,u))
    J = [i + k for i in I]
    
    if linear_ind == True: # Return Linear indices
        I = np.ravel_multi_index([I,J], [m,n])
        return(I)
    return([I,J])
    # end # idiag
