import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t, norm
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Lasso, LassoCV
from slope.solvers import scaled_slope

def forward_stepwise_bic(X, y):
    """
    Forward stepwise selection using BIC as the criterion.
    """
    n, p = X.shape
    selected = []
    current_bic = np.inf
    
    while len(selected) < p:
        best_candidate = None
        best_bic = np.inf
        
        for j in range(p):
            if j in selected:
                continue
            candidates = selected + [j]
            X_cand = X[:, candidates]
            # OLS fit using pseudoinverse for stability
            beta = np.linalg.pinv(X_cand).dot(y)
            rss = np.sum((y - X_cand.dot(beta))**2)
            # BIC = n * ln(RSS/n) + k * ln(n)
            k_param = len(candidates)
            bic = n * np.log(rss / n) + k_param * np.log(n)
            
            if bic < best_bic:
                best_bic = bic
                best_candidate = j
                
        if best_bic < current_bic:
            current_bic = best_bic
            selected.append(best_candidate)
        else:
            break
            
    # Fit final OLS on selected
    beta_ols = np.zeros(p)
    if len(selected) > 0:
        X_sel = X[:, selected]
        beta_ols_sel = np.linalg.pinv(X_sel).dot(y)
        for idx, col in enumerate(selected):
            beta_ols[col] = beta_ols_sel[idx]
            
    return selected, beta_ols

def run_real_data_analysis(q=0.05, alpha=0.05, seed=42):
    """
    Real data analysis comparing SLOPE and competing methods on the Diabetes dataset.
    """
    np.random.seed(seed)
    
    # 1. Load and prepare dataset
    diabetes = load_diabetes()
    X_raw, y_raw = diabetes.data, diabetes.target
    
    # Expand features using polynomial features (interactions only, no bias)
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_poly = poly.fit_transform(X_raw)
    
    # Standardize features and center response
    scaler = StandardScaler()
    X = scaler.fit_transform(X_poly)
    y = y_raw - np.mean(y_raw)
    
    n, p = X.shape
    print(f"Real Data Analysis: n = {n}, p = {p}")
    
    # Estimate noise standard deviation sigma from full OLS model
    beta_full = np.linalg.pinv(X).dot(y)
    rss_full = np.sum((y - X.dot(beta_full))**2)
    sigma_full = np.sqrt(rss_full / (n - p - 1))
    print(f"OLS Full Model Residual Standard Error (sigma_est): {sigma_full:.4f}")
    
    # Store coefficient estimates for each method
    coeff_estimates = {}
    
    # --- 1. SLOPE (scaled SLOPE because sigma is unknown) ---
    print("Fitting Scaled SLOPE...")
    beta_slope, sigma_slope_est = scaled_slope(X, y, q=q, design_type='gaussian')
    S_slope = np.where(np.abs(beta_slope) > 1e-4)[0]
    
    # De-bias SLOPE using OLS
    beta_slope_debiased = np.zeros(p)
    if len(S_slope) > 0:
        X_S = X[:, S_slope]
        beta_ols = np.linalg.pinv(X_S).dot(y)
        beta_slope_debiased[S_slope] = beta_ols
    coeff_estimates['SLOPE'] = beta_slope_debiased
    
    # --- 2. Lasso-Bonferroni (de-biased) ---
    print("Fitting Lasso-Bonferroni...")
    lmbda_bonf = sigma_full * norm.ppf(1.0 - alpha / (2.0 * p))
    lasso_bonf = Lasso(alpha=lmbda_bonf / n, fit_intercept=False, max_iter=5000)
    lasso_bonf.fit(X, y)
    S_bonf = np.where(np.abs(lasso_bonf.coef_) > 1e-4)[0]
    
    beta_bonf_debiased = np.zeros(p)
    if len(S_bonf) > 0:
        X_S = X[:, S_bonf]
        beta_ols = np.linalg.pinv(X_S).dot(y)
        beta_bonf_debiased[S_bonf] = beta_ols
    coeff_estimates['Lasso-Bonf'] = beta_bonf_debiased
    
    # --- 3. Lasso-CV ---
    print("Fitting Lasso-CV...")
    lasso_cv = LassoCV(cv=10, fit_intercept=False, max_iter=5000, n_jobs=-1)
    lasso_cv.fit(X, y)
    coeff_estimates['Lasso-CV'] = lasso_cv.coef_
    
    # --- 4. Stepwise BIC ---
    print("Fitting Stepwise BIC...")
    S_bic, beta_bic = forward_stepwise_bic(X, y)
    coeff_estimates['Stepwise-BIC'] = beta_bic
    
    # --- 5. Marginal BH ---
    print("Fitting Marginal BH...")
    p_vals_marginal = []
    beta_marginal_raw = []
    for j in range(p):
        X_j = X[:, j]
        beta_j = np.dot(X_j, y) / np.dot(X_j, X_j)
        rss = np.sum((y - X_j * beta_j)**2)
        se = np.sqrt(rss / (n - 2)) / np.linalg.norm(X_j)
        t_stat = beta_j / se
        p_val = 2.0 * (1.0 - t.cdf(np.abs(t_stat), df=n-2))
        p_vals_marginal.append(p_val)
        beta_marginal_raw.append(beta_j)
        
    # Apply BH
    p_vals_marginal = np.array(p_vals_marginal)
    sort_idx = np.argsort(p_vals_marginal)
    sorted_p = p_vals_marginal[sort_idx]
    
    i_bh = 0
    for idx in range(p):
        if sorted_p[idx] <= (idx + 1) * q / p:
            i_bh = idx + 1
            
    S_marginal = sort_idx[:i_bh]
    beta_marginal = np.zeros(p)
    if len(S_marginal) > 0:
        X_S = X[:, S_marginal]
        beta_ols = np.linalg.pinv(X_S).dot(y)
        beta_marginal[S_marginal] = beta_ols
    coeff_estimates['Marginal-BH'] = beta_marginal
    
    # --- 6. Full-model BH ---
    print("Fitting Full-model BH...")
    # Full OLS coefficients
    beta_ols_full = np.linalg.pinv(X).dot(y)
    cov_beta = rss_full / (n - p - 1) * np.linalg.pinv(X.T.dot(X))
    
    p_vals_full = []
    for j in range(p):
        se_j = np.sqrt(cov_beta[j, j])
        t_stat = beta_ols_full[j] / se_j
        p_val = 2.0 * (1.0 - t.cdf(np.abs(t_stat), df=n-p-1))
        p_vals_full.append(p_val)
        
    p_vals_full = np.array(p_vals_full)
    sort_idx = np.argsort(p_vals_full)
    sorted_p = p_vals_full[sort_idx]
    
    i_bh = 0
    for idx in range(p):
        if sorted_p[idx] <= (idx + 1) * q / p:
            i_bh = idx + 1
            
    S_full_bh = sort_idx[:i_bh]
    beta_full_bh = np.zeros(p)
    if len(S_full_bh) > 0:
        X_S = X[:, S_full_bh]
        beta_ols = np.linalg.pinv(X_S).dot(y)
        beta_full_bh[S_full_bh] = beta_ols
    coeff_estimates['Full-model-BH'] = beta_full_bh
    
    # 3. Compute summary statistics
    y_var = np.var(y)
    summary_data = []
    
    for name, beta_est in coeff_estimates.items():
        support = np.where(np.abs(beta_est) > 1e-4)[0]
        support_size = len(support)
        
        # Calculate R-squared
        y_pred = X.dot(beta_est)
        rss = np.sum((y - y_pred)**2)
        r2 = 1.0 - (rss / np.sum(y**2))
        
        summary_data.append({
            'Method': name,
            'Selected Variables': support_size,
            'R-squared': r2,
            'Noise Sigma Est': sigma_slope_est if name == 'SLOPE' else (sigma_full if name in ['Lasso-Bonf', 'Full-model-BH'] else np.nan)
        })
        
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('tables/real_data_summary.csv', index=False)
    print("\nSummary Table:")
    print(summary_df.to_string(index=False))
    
    # 4. Save visualization of coefficients
    plt.figure(figsize=(14, 7))
    x_indices = np.arange(p)
    
    colors = {
        'SLOPE': '#1f77b4',
        'Lasso-Bonf': '#d62728',
        'Lasso-CV': '#ff7f0e',
        'Stepwise-BIC': '#2ca02c',
        'Marginal-BH': '#9467bd',
        'Full-model-BH': '#8c564b'
    }
    
    markers = {
        'SLOPE': 'o',
        'Lasso-Bonf': '^',
        'Lasso-CV': 's',
        'Stepwise-BIC': 'D',
        'Marginal-BH': 'v',
        'Full-model-BH': '*'
    }
    
    for name, beta_est in coeff_estimates.items():
        # Plot only non-zero coefficients
        non_zero = np.where(np.abs(beta_est) > 1e-4)[0]
        plt.scatter(non_zero, beta_est[non_zero], label=name, color=colors[name], marker=markers[name], s=80, alpha=0.8)
        
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.title('Estimated Regression Coefficients across Competing Methods (Diabetes Interactions)')
    plt.xlabel('Feature Index')
    plt.ylabel('Estimated Coefficient (De-biased/Shrunk)')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/real_data_coefs.png', dpi=300)
    plt.close()
    
    print("Real data analysis complete. Results saved to 'tables/real_data_summary.csv' and plot to 'figures/real_data_coefs.png'.")
    return coeff_estimates, summary_df

if __name__ == '__main__':
    run_real_data_analysis()
