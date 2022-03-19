import numpy as np
import numpy as np
import scipy.sparse as sps
import numpy.matlib as npm
import DuplicationM as dm


"""
function [A1,b1,A2,b2,mat_obj] = laplacian_constraint_vech(N)
% constraints        
% mat_cons1*L == zeros(N,1)
% mat_cons2*L <= 0
% vec_cons3*L == N

% %% matrix for constraint 1 (zero row-sum)
% for i = 1:N
%     tmp0{i} = sparse(1,N+1-i);
% end
% 
% mat_cons1 = sparse(N,N*(N+1)/2);
% 
% for i = 1:N
%     
%     tmp = tmp0;
%     tmp{i} = tmp{i}+1;
%     for j = 1:i-1
%         tmp{j}(i+1-j) = 1;
%     end
%     
%     mat_cons1(i,:) = horzcat(tmp{:});
%     
% end
% 
% % for i = 1:N
% %     mat_cons1(i,N*i-N+i-(i*(i-1)/2):N*i-(i*(i-1)/2)) = ones(1,N-i+1);
% % end
% % 
% % for i = 1:N-1
% %     xidx = i+1:N;
% %     yidx = i*(N+N-(i-1))/2-(N-i-1):i*(N+N-(i-1))/2-(N-i-1)+N-i-1;
% %     mat_cons1(sub2ind(size(mat_cons1),xidx,yidx)) = 1;
% % end
% 
% %% matrix for constraint 2 (non-positive off-diagonal entries)
% for i = 1:N
%     tmp{i} = ones(1,N+1-i);
%     tmp{i}(1) = 0;
% end
% 
% mat_cons2 = spdiags(horzcat(tmp{:})',0,N*(N+1)/2,N*(N+1)/2);
% 
% %% vector for constraint 3 (trace constraint)
% vec_cons3 = sparse(ones(1,N*(N+1)/2)-horzcat(tmp{:}));
% 
% %% matrix for objective
% % mat_obj = sparse(N^2,N*(N+1)/2);
% % 
% % for i = 1:N
% %     for j = 1:N
% %         if j <= i-1
% %             tmp = tmp0;
% %             tmp{j}(i+1-j) = 1;
% %             mat_obj((i-1)*N+j,:) = horzcat(tmp{:});
% %         else
% %             tmp = tmp0;
% %             tmp{i}(j-i+1) = 1;
% %             mat_obj((i-1)*N+j,:) = horzcat(tmp{:});
% %         end
% %     end
% % end
% 
% mat_obj = vech2vec(N);
% 
% %% create constraint matrices
% % equality constraint A2*vech(L)==b2
% A1 = [mat_cons1;vec_cons3];
% b1 = [sparse(N,1);N];
% 
% % inequality constraint A1*vech(L)<=b1
% A2 = mat_cons2;
% b2 = sparse(N*(N+1)/2,1);
"""

def laplacian_constraint_vech(N):  
    # matrix for objective (vech -> vec)
    mat_obj = dm.DuplicationM(N)
    
    # Matrix Constraint 1
    X = np.ones([N,N])
    r, c = X.shape
    i = range(r*c)
    j = np.matlib.repmat(range(c), r, 1)
    B = sps.csr_matrix((X.flatten('F'), (i, j.flatten('F'))), dtype = np.int_)
    B = B.transpose()
    mat_cons1 = B@mat_obj
    
    # Matrix Constraint 2 Positive off diagonal entries
    Tmp = []
    for i in range(N):
        tmp = [1]*(N-i)
        tmp[0] = 0
        Tmp = Tmp + tmp
    mat_cons2 = sps.spdiags(Tmp, 0, N*(N+1)//2, N*(N+1)//2)
    
    # Vector for constraint 3 (trace constraint)
    arg1 = [1 - i for i in Tmp]
    vec_cons3 = sps.csr_matrix(arg1, dtype = np.int_)
    
    # Create Constraint Matrices
    # Equality constraint A2*vech(L) == b2
    
    A1 = sps.vstack([mat_cons1, vec_cons3])
    b1 = sps.vstack([sps.csr_matrix((N,1), dtype = np.int_), 
                     sps.csr_matrix(([N], ([0],[0])))])
   
    # inequality constraint A1*vech(L)<=b1
    A2 = mat_cons2;
    b2 = sps.csr_matrix((N*(N+1)//2, 1), dtype = np.int_)
    
    return([A1,b1,A2,b2,mat_obj])
    