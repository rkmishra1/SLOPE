import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from slope.solvers import fista_slope, lambda_bh

def make_equicorrelated_design_inv_sqrt(p, rho):
    """
    Computes Sigma^{-1/2} analytically for a p x p equicorrelated covariance matrix:
    Sigma = (1 - rho)*I + rho*1*1^T
    """
    c1 = 1.0 / np.sqrt(1.0 - rho)
    c2 = (1.0 / np.sqrt(1.0 + (p - 1.0) * rho) - c1) / p
    Sigma_inv_sqrt = c1 * np.eye(p) + c2 * np.ones((p, p))
    return Sigma_inv_sqrt

def run_simulation_2(p=1000, num_reps=50, q=0.1, seed=42):
    """
    Simulation 2 (Section 3.1 in Bogdan et al., 2015).
    p: number of hypotheses
    num_reps: number of Monte Carlo replicates
    q: target FDR level
    """
    np.random.seed(seed)
    k_values = [0, 10, 20, 30, 40, 50, 60, 70, 80]
    
    # Lab variance components
    sigma_tau_sq = 2.5
    sigma_z_sq = 2.5
    m = 5 # Number of labs
    
    # Standard deviation of average score y_bar
    # sigma = sqrt( (sigma_tau_sq + sigma_z_sq) / m ) = 1.0
    sigma = np.sqrt((sigma_tau_sq + sigma_z_sq) / m)
    rho = (sigma_tau_sq / m) / (sigma**2) # rho = 0.5
    
    # 1. Compute Sigma_inv_sqrt and standardized design matrix X
    Sigma_inv_sqrt = make_equicorrelated_design_inv_sqrt(p, rho)
    # Norm of columns of Sigma_inv_sqrt (all are identical due to symmetry)
    c = np.linalg.norm(Sigma_inv_sqrt[:, 0])
    X = Sigma_inv_sqrt / c
    
    # Nonzero signal value for mu
    signal_val_mu = np.sqrt(2.0 * np.log(p)) / c
    
    results = {
        'SLOPE': {'fdr': [], 'power': []},
        'Marginal-BH': {'fdr': [], 'power': []}
    }
    
    # Pre-generate lambda sequence for SLOPE (sigma = 1.0 for whitened noise)
    l_bh = lambda_bh(p, q, sigma=1.0)
    
    for k in k_values:
        print(f"Simulation 2: Running for k = {k}...")
        
        rep_metrics = {
            'SLOPE': {'fdp': [], 'tp_rate': []},
            'Marginal-BH': {'fdp': [], 'tp_rate': []}
        }
        
        for rep in range(num_reps):
            # Generate mu
            mu_true = np.zeros(p)
            S_true = set()
            if k > 0:
                active_idx = np.random.choice(p, size=k, replace=False)
                mu_true[active_idx] = signal_val_mu
                S_true = set(active_idx)
                
            # Generate lab effects tau ~ N(0, sigma_tau_sq)
            tau = np.random.normal(0, np.sqrt(sigma_tau_sq), size=m)
            
            # Generate lab-specific observations: y_{i,j} = mu_i + tau_j + z_{i,j}
            # and average them: y_bar_i = mu_i + tau_bar + z_bar_i
            tau_bar = np.mean(tau)
            z_bar = np.random.normal(0, np.sqrt(sigma_z_sq / m), size=p)
            y_bar = mu_true + tau_bar + z_bar
            
            # --- competitor 1: Marginal BH ---
            # Two-sided p-values under H0: mu_i = 0 with marginal variance sigma^2 = 1.0
            p_values = 2.0 * (1.0 - norm.cdf(np.abs(y_bar) / sigma))
            
            # Run BH step-up manually
            sort_idx = np.argsort(p_values)
            sorted_p = p_values[sort_idx]
            
            # Find iBH = max { i : p_(i) <= i * q / p }
            i_bh = 0
            for idx in range(p):
                # idx is 0-based, corresponding to i = idx + 1
                if sorted_p[idx] <= (idx + 1) * q / p:
                    i_bh = idx + 1
                    
            S_marginal = set(sort_idx[:i_bh])
            
            # --- competitor 2: Whitening + SLOPE ---
            # Whiten the noise
            y_tilde = Sigma_inv_sqrt.dot(y_bar)
            
            # Fit SLOPE on standardized X
            beta_slope_est = fista_slope(X, y_tilde, l_bh)
            S_slope = set(np.where(np.abs(beta_slope_est) > 1e-4)[0])
            
            # Evaluate metrics
            for name, S_est in [('SLOPE', S_slope), ('Marginal-BH', S_marginal)]:
                # FDP
                fdp = len(S_est - S_true) / max(1.0, len(S_est))
                rep_metrics[name]['fdp'].append(fdp)
                
                # Power
                if k > 0:
                    tp_rate = len(S_est & S_true) / k
                    rep_metrics[name]['tp_rate'].append(tp_rate)
                else:
                    rep_metrics[name]['tp_rate'].append(0.0)
                    
        for name in results.keys():
            results[name]['fdr'].append(np.mean(rep_metrics[name]['fdp']))
            results[name]['power'].append(np.mean(rep_metrics[name]['tp_rate']))
            
    # Save the plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = {'SLOPE': '#1f77b4', 'Marginal-BH': '#ff7f0e'}
    markers = {'SLOPE': 'o', 'Marginal-BH': 's'}
    
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
    
    plt.tight_layout()
    plt.savefig('figures/sim2_correlated.png', dpi=300)
    plt.close()
    
    print("Simulation 2 complete. Figure saved to 'figures/sim2_correlated.png'.")
    return k_values, results

if __name__ == '__main__':
    run_simulation_2(num_reps=5)
