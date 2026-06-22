# =============================================================================
# Unless this comment is removed, this code is meant solely for members of Liam
# McAllister's group or other people associated with it.
# =============================================================================
#
# -----------------------------------------------------------------------------
# Description:  Contains functons to perform lattice computations
# -----------------------------------------------------------------------------

# standard imports
import time, warnings
import math
import itertools
from flint import fmpz, fmpz_mat
import numpy as np
import sympy as smp

# CYTools imports
# (for utilities, there should be ~no such imports...)

# repo imports

warnings.warn("AS: Check where these functions are being used and remove them accordingly!")

# ADD SECTIONS, DESCRIPTIONS
# --------------------------
def extended_euclidean(w):
    r"""
    
    **Description:**
    Computes the GCD and Bézout's identity using the extended euclidean algorithm. Also computes an integer
    basis transformation Lambda that brings the input array to form (gcd,0,...,0).
    
    Args:
        w (numpy array): Input array.

    Returns:
        tuple (tuple): (Bezout,GCD,Lambda)
        
    """
    
    w = np.asarray(w)
    nonvan_flag = (w!=0)
    van_flag = (w==0)
    nonvan_pos = np.where(nonvan_flag)[0]
    van_pos = np.where(van_flag)[0]

    Bezout = np.zeros(len(w),dtype=int)

    if sum(nonvan_flag)==1:
        GCD = w[nonvan_flag][0]
        Bezout[nonvan_flag] = 1
        Lambda_final = np.identity(len(w),dtype=int)
        Lambda_final[0][0]=0
        Lambda_final[nonvan_pos[0]][nonvan_pos[0]]=0
        Lambda_final[0][nonvan_pos[0]]=1
        Lambda_final[nonvan_pos[0]][0]=1
        
    else:
        v = w[nonvan_flag]

        acoeff = np.abs(v)
        reordering = np.flip(np.argsort(acoeff))
        acoeffsorted = acoeff[reordering]
        Lambda = np.array([np.eye(1,len(reordering),i,dtype=int)[0] for i in reordering])

        dim_red = 0
        while True:
            divs = acoeffsorted[:-1]/acoeffsorted[-1]
            qs = divs.astype(int)
            rs = np.rint(((divs-qs)*acoeffsorted[-1])).astype(int)+np.arange(len(divs))*1e-10
            rssorted = np.flip(np.sort(rs))
  
            perm = np.array([i==rs for i in rssorted],dtype=int)
            acoeffsorted = np.rint(np.concatenate(([acoeffsorted[-1]],rssorted))).astype(int)
            LambdaNext0 = np.block([[qs,np.transpose([[1]])],[perm,np.transpose([np.zeros(len(perm))])]]).astype(int)
            LambdaNext = np.block([[LambdaNext0,np.zeros([len(LambdaNext0),dim_red])],[np.zeros([dim_red,len(LambdaNext0)]),np.identity(dim_red)]])
            Lambda = LambdaNext@Lambda
            
            posnonvan = np.where(acoeffsorted>0)[0]
            posvan = np.where(acoeffsorted==0)[0]
            acoeffsorted = acoeffsorted[posnonvan]
            dim_red = dim_red+len(posvan)

            if len(acoeffsorted)==1:
                break

        Bezout0 = (np.rint(np.transpose(np.linalg.inv(Lambda))[0])*np.sign(v)).astype(int)
        Lambda0 = (np.rint(np.transpose(np.linalg.inv(Lambda)))*np.sign(v)).astype(int)
        Lambda_tilde = np.block([[np.zeros([len(Lambda0),len(w)-len(Lambda0)],dtype=int),Lambda0]
                                 ,[np.identity(len(w)-len(Lambda0),dtype=int),np.zeros([len(w)-len(Lambda0),len(Lambda0)],dtype=int)]])
        
        
        Lambda_final = np.identity(len(w),dtype=int)
        Lambda_final[nonvan_pos] = Lambda_tilde.T[len(w)-len(Lambda0):len(Lambda_tilde)]
        Lambda_final[van_pos] = Lambda_tilde.T[0:len(w)-len(Lambda0)]
        Lambda_final = Lambda_final.T
        
        GCD = np.rint(sum(Bezout0*v)).astype(int)

        Bezout[nonvan_flag] = Bezout0
        
    return (Bezout,GCD,Lambda_final)

def norm_wrt_metric(v,Z):
    r"""
    **Description:**
    Computes vector norm with respect to given input metric Z.

    Args:
        v (numpy array): Input vector.
        Z (numpy array): Symmetric positive definite metric.

    Returns:
        _ (float): Norm of the input vector with respect to the metric.
    """
    return v@Z@v

def scalar_projection_wrt_metric(v,Z,w):
    r"""
    **Description:**
    Computes the projection of one vector on another with respect to given input metric Z.

    Args:
        v (numpy array): Input vector.
        Z (numpy array): Symmetric positive definite metric.
        w (numpy array): Input vector to be projected on.

    Returns:
        _ (float): Projection of v on w.
    """
    
    return v@Z@w/norm_wrt_metric(w,Z)

def GramSchmidt_wrt_metric(b,Z):
    r"""
    **Description:**
    Applies step one of the Gram-Schmidt process to the rows of b, i.e. returns a set of orthogonal vectors 
    w.r.t. the metric Z

    Args:
        b (numpy array): input array. The first step of the Gram-Schmidt process is applied to its rows.
        Z (numpy array): symmetric positive definite metric.

    Returns:
        numpy array: array of mutually orthogonal vectors obtained via the Gram-Schmidt process.
    """
    btilde = [b[0]]
    for i in range(1,len(b)):
        btilde.append(b[i]-sum(map(lambda y: scalar_projection_wrt_metric(b[i],Z,y)*y,btilde)))
        
    return np.array(btilde)

def is_LLL_reduced_wrt_metric(b,Z,delta=3/4,tolerance=1e-8):
    r"""
    
    **Description:**
    Checks whether or not a set of vectors is already LLL reduced with respect to the metric Z.

    Args:
        b (numpy array): input integer array whose rows are interpreted as a lattice basis.
        Z (numpy array): symmetric positive definite metric.
        delta (float, optional): delta-parameter of the LLL reduction. Must choose 1/4<delta<1. Default value is 3/4.
        tolerance (float,optional): Tolerance used in the LLL reduction. Defaults to c:var:`tolerance=1e-8`.

    Returns:
        boolean: True if the lattice basis b is already LLL-reduced.
    """
    
    # Perform Gram-Schmidt
    btilde = GramSchmidt_wrt_metric(b,Z)
    
    # Comput mu as the matrix of scalar projections of the old vectors on the new ones
    prod1 = Z@btilde.T
    prod2 = btilde@prod1
    mu = b@prod1/np.diagonal(prod2)
    
    # Test if projections <=0.5
    test1 = np.all(np.tril(mu,-1)<=0.5+1e-5)
    
    bmod = (np.diag(mu,k=-1)*btilde[:-1].T).T+btilde[1:]
    prod3 = bmod@Z@bmod.T
    
    test2 = np.all(delta*np.diag(prod2)[:-1]-np.diag(prod3)<=tolerance)
    
    return (test1 and test2)

def LLL_reduction_wrt_metric(b,Z,delta=3/4,tolerance=1e-8,max_iterations=100):
    r"""
    **Description:**
    Computes the LLL reduction of the lattice basis b, with respect to the positive definite metric Z. Wrapper for
    LLL_reduction_wrt_metric_raw making the algorithm faster.

    Args:
        b (numpy array): input integer array whose rows are interpreted as a lattice basis.
        Z (numpy array): symmetric positive definite metric.
        delta (float, optional): delta-parameter of the LLL reduction. Must choose 1/4<delta<1. Default value is 3/4.
        tolerance (float,optional): Tolerance used in the LLL reduction. Defaults to c:var:`tolerance=1e-8`.
        max_iterations (int, optional): Maximum number of iterations used for the LLL reduction.

    Returns:
        numpy array: LLL reduced lattice basis
    """
    warnings.warn("What is b = LLL_red(b) doing? Why do we need to do it again inside LLL_reduction_wrt_metric_raw?")
    
    b = LLL_red(b)
    
    rounding_factor = np.max(np.linalg.inv(np.linalg.cholesky(Z)))
    while True:
        e = np.rint(rounding_factor*np.linalg.cholesky(Z)).astype(int)
        rounding_factor = rounding_factor+1
        if np.linalg.det(e)!=0:
            break
    
    # Compute updated metric
    Ztilde = np.linalg.inv(e)@Z@np.linalg.inv(e.T)
    
    # Compute updated generators
    btilde = b@e
    
    # Peform LLL reduction wrt to `Ztilde``
    btilde_new = LLL_reduction_wrt_metric_raw(btilde,Ztilde,delta=delta)
    
    # Compute basis transformation using the LLL reduced generators
    basis_trafo = np.rint(btilde_new@np.linalg.inv(btilde)).astype(int)
    
    # Compute the basis generators in new basis
    bnew = basis_trafo@b
    
    # Return output
    return bnew
    

def LLL_reduction_wrt_metric_raw(b,Z,delta=3/4,tolerance=1e-8,max_iterations=100):
    r"""
    **Description:**
    Computes the LLL reduction of the lattice basis b, with respect to the positive definite metric Z.

    Args:
        b (numpy array): input integer array whose rows are interpreted as a lattice basis.
        Z (numpy array): symmetric positive definite metric.
        delta (float, optional): delta-parameter of the LLL reduction. Must choose 1/4<delta<1. Default value is 3/4.
        tolerance (float,optional): Tolerance used in the LLL reduction. Defaults to c:var:`tolerance=1e-8`.
        max_iterations (int, optional): Maximum number of iterations used for the LLL reduction.

    Returns:
        numpy array: LLL reduced lattice basis
    """
    
    b = LLL_red(b)
    
    # Check if basis is already LLL reduced
    isLLLreduced = is_LLL_reduced_wrt_metric(b,Z,delta,tolerance=tolerance)
    
    # Introduce counter to avoid crash on redcloud
    warnings.warn("AS: might need to be updated in the future.")
    count=0
    while (not(isLLLreduced) and count < max_iterations):
        
        # Perform Gram-Schmidt orthogonalisation wrt metric Z
        btilde = GramSchmidt_wrt_metric(b,Z)
        
        # Loop of generators
        for i in range(1,len(b)):
            for k in range(i):
                j = i-1-k
                b[i] = b[i] - np.rint(scalar_projection_wrt_metric(b[i],Z,btilde[j])).astype(int)*b[j]
                btilde = GramSchmidt_wrt_metric(b,Z)
                
        
        # Comput mu as the matrix of scalar projections of the old vectors on the new ones
        prod1 = Z@btilde.T
        prod2 = btilde@prod1
        mu = b@prod1/np.diagonal(prod2)
        
        # Test where we need to swap generators
        bmod = (np.diag(mu,k=-1)*btilde[:-1].T).T+btilde[1:]
        prod3 = bmod@Z@bmod.T
        swappos = np.where(delta*np.diag(prod2)[:-1]-np.diag(prod3)>0)[0]
        
        # If necessary, swap generators
        if len(swappos)>0:
            b[[swappos[0]+1, swappos[0]]] = b[[swappos[0], swappos[0]+1]]
            
        # Test if LLL reduction successful
        isLLLreduced = is_LLL_reduced_wrt_metric(b,Z,delta,tolerance=tolerance)
        count+=1
        
    # Return warning if we required too many iterations
    if count==max_iterations:
        warnings.warn("Maximum number of iterations achieved!")
        
    # Return results
    return b
    
def LLL_red(v):
    r"""
    **Description:**
    Returns LLL reduced lattice basis of lattice generated by v. Just a wrapper for fmpz_mat(v).lll()

    Args:
        v (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    v = np.array(v)
    if len(v)==0:
        return v
    else:
        lll0 = np.array(fmpz_mat(v.tolist()).lll().tolist()).astype(int)
        return np.delete(lll0,np.where(np.all(lll0==0,axis=1))[0],0)

def orthogonal_lattice(gens_in):
    r"""
    
    **Description:**
    Returns the generators of the lattice orthogonal to the lattice generated by gens_in.

    Args:
        gens_in (_type_): _description_

    Returns:
        _type_: _description_
    """
    gens = np.array(gens_in)
    d = len(gens)
    n = len(gens[0])
    exponent = (n-1)/2 + (n-d)*(n-d-1)/4
    c = int(np.ceil((2**exponent)*np.prod([np.linalg.norm(g) for g in gens])))
    b_T = np.concatenate((c*gens,np.identity(n, dtype=int)))
    b_T_mat = fmpz_mat(b_T.T.tolist())
    b_T_lll = [[int(ii) for ii in i][-n:] for i in np.array(b_T_mat.lll().tolist(), dtype=int)[:n-d]]
    return b_T_lll
    


def lattice_dual(m_in):
    """
    
    **Description:**
    Given a lattice generated by m_in, returns the generators of the dual lattice.

    Args:
        m_in (_type_): _description_

    Returns:
        _type_: _description_
    """
    m = smp.Matrix(m_in)
    return m*(m.T*m).inv()

def integral_inner_product_generators(m_in):
    """
    
    **Description:**
    Given a list of vectors m_in, returns the generators of the integral lattice whose elements dot to Z with all vectors in m_in.

    Args:
        m_in (_type_): _description_

    Returns:
        _type_: _description_
    """
    m = smp.Matrix(m_in)
    m_scaling = np.lcm.reduce([ii.q for i in np.array(m) for ii in i if type(ii) is not int])
    m_scaled = m_scaling*np.array(m)
    m_scaled_f = fmpz_mat([[fmpz(int(ii)) for ii in i] for i in m_scaled])
    m_scaled_hnf_f = m_scaled_f.hnf()
    m_dual_hnf = (1/m_scaling)*np.array([[int(ii) for ii in i] for i in m_scaled_hnf_f.tolist()])
    return np.array([[int(round(rr)) for rr in r] for r in lattice_dual(m_dual_hnf).tolist() if r!=[0]*len(m_in[0])])

