import numpy as np
import numpy_groupies as npg

# function [I J] = itril(sz, k)
# function [I J] = itril(sz, k) # OR
# I = itril(sz, k)
#
# Return the subindices [I J] (or linear indices I if single output call)
# in the purpose of extracting an lower triangular part of the matrix of
# the size SZ. Input k is optional shifting. For k=0, extract from the main
# diagonal. For k>0 -> above the diagonal, k<0 -> below the diagonal
# 
# This returnd same as [...] = find(tril(ones(sz),k))
# - Output is a column and sorted with respect to linear indice
# - No intermediate matrix is generated, that could be useful for large
#   size problem
# - Mathematically, A(itril(size(A)) is called (lower) "half-vectorization"
#   of A 
#
# Example:
#
# A = [ 7     5     4
#       4     2     3
#       9     1     9
#       3     5     7 ]
#
# I = itril(size(A))  # gives [1 2 3 4 6 7 8 11 12]'
# A(I)                # gives [7 4 9 3 2 1 5  9  7]' OR A(tril(A)>0)
#
# Author: Bruno Luong <brunoluong@yahoo.com>
# Date: 21/March/2009

def itril(sz, k = 0, linear_ind = True):
    
    if np.isscalar(sz):
        sz = [sz, sz]
    m = sz[0]
    n = sz[1]
    
    # Main Diagonal by default
    nc = n - max(k,0)
    
    lo = [int(i) for i in np.ones(nc)] # lower row indice for each column
    
    hi = [min(i - min(k,0), m) for i in range(1,nc+1)] # upper row indice for each column
    
    if len(lo) == 0: 
        I = []
        J = []
    else:
        temp = [1] + [hi[i] - lo[i] + 1 for i in range(len(hi))] 
        c = np.cumsum(temp) # cumsum of the length
        temp = [0] + hi[:len(hi)-1]
        I = npg.aggregate(c[0:len(c)-1], 
                          [lo[i] - temp[i] - 1 for i in range(len(lo))], 
                          size = c[len(c)-1])[1:]
        
        I = np.cumsum([i + 1 for i in I]) # row indice
        
        J = npg.aggregate(c, 1)[1:]; 
        
        J[0] = 1 + max(k,0) # The row indices starts from this value
        
        J = np.cumsum(J[0:len(J)-1]) # column indice
        
        # print('J after cummulation is ' + str(J))
    # end
    
    I = [i - 1 for i in I]
    J = [j - 1 for j in J]
    if linear_ind == True: # Convert to Linear Indices
        I = np.ravel_multi_index([J,I], [m,n])
        return(I)
    return([J,I])
