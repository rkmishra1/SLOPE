import numpy as np
from scipy.stats import norm

def fast_prox_sl1(y, lmbda):
    """
    Stack-based algorithm for FastProxSL1 (Algorithm 4 in Bogdan et al., 2015).
    Inputs:
        y: non-negative, non-increasing 1D numpy array
        lmbda: non-negative, non-increasing 1D numpy array of same shape
    Returns:
        x: non-negative, non-increasing 1D numpy array of same shape
    """
    n = len(y)
    # The stack will hold blocks represented as dicts:
    # 'i': start index of block (0-based)
    # 'j': end index of block (0-based)
    # 's': sum of (y_k - lmbda_k) for elements in block
    # 'w': thresholded average: max(0.0, sum / size)
    stack = []
    
    for k in range(n):
        val = y[k] - lmbda[k]
        block = {
            'i': k,
            'j': k,
            's': val,
            'w': max(0.0, val)
        }
        stack.append(block)
        
        # Merge while the stack has more than 1 block and monotonicity is violated:
        # w_{t-1} <= w_t (using 0-based indexing: stack[-2]['w'] <= stack[-1]['w'])
        while len(stack) > 1 and stack[-2]['w'] <= stack[-1]['w']:
            b_top = stack.pop()
            b_prev = stack[-1]
            
            # Merge top into prev
            b_prev['j'] = b_top['j']
            b_prev['s'] = b_prev['s'] + b_top['s']
            size = b_prev['j'] - b_prev['i'] + 1
            b_prev['w'] = max(0.0, b_prev['s'] / size)
            
    # Reconstruct x from stack blocks
    x = np.zeros(n)
    for block in stack:
        x[block['i']:block['j']+1] = block['w']
        
    return x

def prox_sorted_l1(v, lmbda):
    """
    Computes the proximal operator of the sorted L1 norm of v with weights lmbda.
    prox_J(v) = argmin_x 0.5 * ||v - x||_2^2 + sum(lmbda_i * |x|_(i))
    """
    v = np.asarray(v)
    lmbda = np.asarray(lmbda)
    
    # Save signs and take absolute values
    s = np.sign(v)
    y = np.abs(v)
    
    # Sort y in descending order, keeping track of the permutation
    sort_idx = np.argsort(y)[::-1]
    y_sorted = y[sort_idx]
    
    # Run the stack-based FastProxSL1 algorithm
    x_sorted = fast_prox_sl1(y_sorted, lmbda)
    
    # Restore the original order (unsort)
    unsort_idx = np.argsort(sort_idx)
    x = x_sorted[unsort_idx]
    
    # Restore the signs
    res = x * s
    return res

def fista_slope(X, y, lmbda, max_iter=1000, tol=1e-6):
    """
    FISTA algorithm for SLOPE (Algorithm 2 in Bogdan et al., 2015).
    X: design matrix (n x p)
    y: response vector (n)
    lmbda: sorted penalization sequence (p)
    """
    X = np.asarray(X)
    y = np.asarray(y)
    lmbda = np.asarray(lmbda)
    
    n, p = X.shape
    
    # Compute Lipschitz constant L = ||X||_2^2
    L = np.linalg.norm(X, ord=2) ** 2
    
    # Initialize variables
    beta = np.zeros(p)
    a = np.zeros(p)
    t_fista = 1.0
    
    for i in range(max_iter):
        # Gradient of the smooth term: X^T (X a - y)
        grad = X.T.dot(X.dot(a) - y)
        v = a - (1.0 / L) * grad
        
        # Proximal step with step size 1/L
        beta_new = prox_sorted_l1(v, lmbda / L)
        
        # Check convergence
        if np.linalg.norm(beta_new - beta, ord=np.inf) < tol:
            beta = beta_new
            break
            
        # Update FISTA parameters
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * (t_fista ** 2)))
        a = beta_new + ((t_fista - 1.0) / t_new) * (beta_new - beta)
        
        beta = beta_new
        t_fista = t_new
        
    return beta

def scaled_slope(X, y, q, design_type='gaussian', max_iter=100, tol=1e-4):
    """
    Iterative SLOPE fitting when sigma is unknown (Algorithm 5 in Bogdan et al., 2015).
    """
    n, p = X.shape
    
    # 1. Compute initial sequence lambda_S for sigma = 1
    if design_type == 'orthogonal':
        l_s = lambda_bh(p, q, sigma=1.0)
    else:
        l_s = lambda_g_star(n, p, q, sigma=1.0)
        
    # Initialize: S = empty set
    S = set()
    beta = np.zeros(p)
    sigma_est = np.std(y) # Initial estimate
    
    for iteration in range(max_iter):
        # Compute RSS by regressing y onto variables in S
        if len(S) == 0:
            rss = np.sum(y**2)
            df = n - 1
        else:
            S_list = list(S)
            X_S = X[:, S_list]
            # Use pseudoinverse for stable OLS
            beta_ols = np.linalg.pinv(X_S).dot(y)
            residuals = y - X_S.dot(beta_ols)
            rss = np.sum(residuals**2)
            df = n - len(S) - 1
            
        sigma_est_new = np.sqrt(rss / max(1.0, df))
        
        # Fit SLOPE with parameter sequence sigma_est * l_s
        lmbda = sigma_est_new * l_s
        beta_new = fista_slope(X, y, lmbda)
        
        S_new = set(np.where(np.abs(beta_new) > 1e-5)[0])
        
        # Check convergence of support set and estimate
        if S_new == S and np.abs(sigma_est_new - sigma_est) < tol:
            beta = beta_new
            sigma_est = sigma_est_new
            break
            
        S = S_new
        beta = beta_new
        sigma_est = sigma_est_new
        
    return beta, sigma_est

def lambda_bh(p, q, sigma=1.0):
    """
    Benjamini-Hochberg critical sequence.
    lambda_i = sigma * Phi^-1(1 - i * q / 2p)
    """
    i = np.arange(1, p + 1)
    return sigma * norm.ppf(1.0 - i * q / (2.0 * p))

def lambda_g_star(n, p, q, sigma=1.0):
    """
    Adjusted sequence lambda_G* for Gaussian designs (Section 3.2.2 in Bogdan et al., 2015).
    """
    l_bh = lambda_bh(p, q, sigma=sigma)
    l_g = np.zeros(p)
    l_g[0] = l_bh[0]
    
    # Compute recursively up to min(p, n-1)
    limit = min(p, n - 1)
    
    sum_sq = l_g[0]**2
    for i in range(1, limit):
        # i in python (0-based) represents element i+1 (1-based)
        # w(i) = 1 / (n - i - 1)
        val = 1.0 + (1.0 / (n - (i + 1))) * sum_sq
        l_g[i] = l_bh[i] * np.sqrt(val)
        sum_sq += l_g[i]**2
        
    # If p >= n, we pad the remaining entries to prevent numerical issues
    if p >= n:
        l_g[limit:] = np.inf
        
    # Find the global minimum k*
    k_star = np.argmin(l_g[:limit+1] if p < n else l_g[:limit])
    val_k_star = l_g[k_star]
    
    l_g_star = np.copy(l_g)
    l_g_star[k_star:] = val_k_star
    
    # Make sure we don't have inf values
    if p >= n:
        l_g_star[limit:] = val_k_star
        
    return l_g_star

def lambda_mc(X, q, sigma=1.0, num_draws=500, max_k=100):
    """
    Computes the Monte Carlo adjusted sequence lambda_MC (Section 3.2.2 in Bogdan et al., 2015).
    """
    n, p = X.shape
    l_bh = lambda_bh(p, q, sigma=sigma)
    l_mc = np.zeros(p)
    l_mc[0] = l_bh[0]
    
    limit = min(p, n - 1, max_k)
    
    for i in range(1, limit):
        u_sq_sum = 0.0
        for _ in range(num_draws):
            # Select S of size i
            S_indices = np.random.choice(p, size=i, replace=False)
            # Select j not in S
            j_candidate = np.random.choice(p)
            while j_candidate in S_indices:
                j_candidate = np.random.choice(p)
                
            X_S = X[:, S_indices]
            X_j = X[:, j_candidate]
            
            l_slice = l_mc[:i]
            
            try:
                h = np.linalg.solve(X_S.T.dot(X_S), l_slice)
                u = X_j.dot(X_S.dot(h))
                u_sq_sum += u**2
            except np.linalg.LinAlgError:
                h = np.linalg.pinv(X_S.T.dot(X_S)).dot(l_slice)
                u = X_j.dot(X_S.dot(h))
                u_sq_sum += u**2
                
        mean_u_sq = u_sq_sum / num_draws
        l_mc[i] = l_bh[i] * np.sqrt(1.0 + mean_u_sq)
        
    k_star = np.argmin(l_mc[:limit])
    val_k_star = l_mc[k_star]
    l_mc[k_star:] = val_k_star
    
    return l_mc
