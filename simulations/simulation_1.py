import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, LassoCV
from scipy.stats import norm
from slope.solvers import fista_slope, lambda_g_star, lambda_bh

def run_simulation_1(n=1000, p=1000, num_reps=50, q=0.1, alpha=0.1, seed=42):
    """
    Simulation 1 (Section 1.3.3 in Bogdan et al., 2015).
    n: number of observations
    p: number of features
    num_reps: number of Monte Carlo replicates
    q: target FDR level for SLOPE
    alpha: target FWER level for Lasso-Bonferroni
    """
    np.random.seed(seed)
    k_values = [0, 10, 20, 30, 40, 50]
    
    # We will collect average metrics for each competitor
    results = {
        'SLOPE': {'fdr': [], 'power': [], 'mse': []},
        'Lasso-Bonf': {'fdr': [], 'power': [], 'mse': []},
        'Lasso-CV': {'fdr': [], 'power': [], 'mse': []}
    }
    
    # Pre-generate lambda sequences (noise standard deviation sigma = 1)
    l_g_star = lambda_g_star(n, p, q, sigma=1.0)
    lmbda_bonf = norm.ppf(1.0 - alpha / (2.0 * p))
    
    # Signal magnitude
    signal_val = np.sqrt(2.0 * np.log(p))
    
    for k in k_values:
        print(f"Simulation 1: Running for k = {k}...")
        
        # Temp storage for replicates
        rep_metrics = {
            'SLOPE': {'fdp': [], 'tp_rate': [], 'rel_mse': []},
            'Lasso-Bonf': {'fdp': [], 'tp_rate': [], 'rel_mse': []},
            'Lasso-CV': {'fdp': [], 'tp_rate': [], 'rel_mse': []}
        }
        
        for rep in range(num_reps):
            # 1. Generate design matrix X (entries i.i.d. N(0, 1/n))
            X = np.random.normal(0, 1.0 / np.sqrt(n), size=(n, p))
            
            # 2. Generate true coefficients beta
            beta_true = np.zeros(p)
            if k > 0:
                # Randomly place active coefficients
                active_idx = np.random.choice(p, size=k, replace=False)
                beta_true[active_idx] = signal_val
                
            # 3. Generate response y = X * beta + z where z ~ N(0, I)
            z = np.random.normal(0, 1.0, size=n)
            y = X.dot(beta_true) + z
            
            # True active indices set
            S_true = set(np.where(beta_true > 0)[0])
            
            # --- competitor 1: SLOPE (de-biased) ---
            beta_slope_est = fista_slope(X, y, l_g_star)
            S_slope = set(np.where(np.abs(beta_slope_est) > 1e-4)[0])
            
            # De-bias
            beta_slope_debiased = np.zeros(p)
            if len(S_slope) > 0:
                X_S = X[:, list(S_slope)]
                beta_ols = np.linalg.pinv(X_S).dot(y)
                for idx, col in enumerate(S_slope):
                    beta_slope_debiased[col] = beta_ols[idx]
                    
            # --- competitor 2: Lasso-Bonferroni (de-biased) ---
            # Lasso with alpha_param = lambda_bonf / n
            lasso_bonf = Lasso(alpha=lmbda_bonf / n, fit_intercept=False, max_iter=2000)
            lasso_bonf.fit(X, y)
            S_bonf = set(np.where(np.abs(lasso_bonf.coef_) > 1e-4)[0])
            
            beta_bonf_debiased = np.zeros(p)
            if len(S_bonf) > 0:
                X_S = X[:, list(S_bonf)]
                beta_ols = np.linalg.pinv(X_S).dot(y)
                for idx, col in enumerate(S_bonf):
                    beta_bonf_debiased[col] = beta_ols[idx]
                    
            # --- competitor 3: Lasso-CV ---
            # LassoCV uses cv=10
            lasso_cv = LassoCV(cv=10, fit_intercept=False, max_iter=2000, n_jobs=-1)
            lasso_cv.fit(X, y)
            beta_cv = lasso_cv.coef_
            S_cv = set(np.where(np.abs(beta_cv) > 1e-4)[0])
            
            # Calculate metrics
            for name, S_est, beta_est in [
                ('SLOPE', S_slope, beta_slope_debiased),
                ('Lasso-Bonf', S_bonf, beta_bonf_debiased),
                ('Lasso-CV', S_cv, beta_cv) # Using lasso_cv.coef_ without de-biasing
            ]:
                # If we are evaluating Lasso-CV, make sure we use beta_cv
                if name == 'Lasso-CV':
                    beta_est = beta_cv
                    
                # False Discovery Proportion
                fdp = len(S_est - S_true) / max(1.0, len(S_est))
                rep_metrics[name]['fdp'].append(fdp)
                
                # Power (TP rate)
                if k > 0:
                    tp_rate = len(S_est & S_true) / k
                    rep_metrics[name]['tp_rate'].append(tp_rate)
                    
                    # Relative MSE: ||X beta_hat - X beta||^2 / ||X beta||^2
                    pred_err = np.sum((X.dot(beta_est) - X.dot(beta_true))**2)
                    true_norm = np.sum((X.dot(beta_true))**2)
                    rep_metrics[name]['rel_mse'].append(pred_err / true_norm)
                else:
                    rep_metrics[name]['tp_rate'].append(0.0)
                    
        # Average replicates
        for name in results.keys():
            results[name]['fdr'].append(np.mean(rep_metrics[name]['fdp']))
            results[name]['power'].append(np.mean(rep_metrics[name]['tp_rate']))
            if k > 0:
                results[name]['mse'].append(np.mean(rep_metrics[name]['rel_mse']))
            else:
                results[name]['mse'].append(0.0)
                
    # Save the plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Set styles
    colors = {'SLOPE': '#1f77b4', 'Lasso-Bonf': '#d62728', 'Lasso-CV': '#ff7f0e'}
    markers = {'SLOPE': 'o', 'Lasso-Bonf': '^', 'Lasso-CV': 's'}
    
    # 1. FDR plot
    ax = axes[0]
    for name in results.keys():
        ax.plot(k_values, results[name]['fdr'], label=name, color=colors[name], marker=markers[name], lw=2)
    ax.axhline(y=q, color='gray', linestyle='--', label='Nominal FDR')
    ax.set_title('False Discovery Rate (FDR)')
    ax.set_xlabel('Number of true signals (k)')
    ax.set_ylabel('FDR')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 2. Power plot
    ax = axes[1]
    for name in results.keys():
        ax.plot(k_values, results[name]['power'], label=name, color=colors[name], marker=markers[name], lw=2)
    ax.set_title('Power')
    ax.set_xlabel('Number of true signals (k)')
    ax.set_ylabel('Power')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 3. Relative MSE plot (for k > 0)
    ax = axes[2]
    for name in results.keys():
        ax.plot(k_values[1:], results[name]['mse'][1:], label=name, color=colors[name], marker=markers[name], lw=2)
    ax.set_title('Relative MSE')
    ax.set_xlabel('Number of true signals (k)')
    ax.set_ylabel('Relative MSE')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('figures/sim1_metrics.png', dpi=300)
    plt.close()
    
    print("Simulation 1 complete. Figure saved to 'figures/sim1_metrics.png'.")
    return k_values, results

if __name__ == '__main__':
    run_simulation_1(num_reps=5)
