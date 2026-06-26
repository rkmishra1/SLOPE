# SLOPE: Sorted L-One Penalized Estimation

Python reference implementation of the **SLOPE** (Sorted L-One Penalized Estimation) regularized regression method, reproducing the simulations and competing methods benchmark from the seminal paper:
> **Bogdan, M., van den Berg, E., Sabatti, C., Su, W., & Candès, E. J. (2015).** *SLOPE—Adaptive Variable Selection via Convex Optimization.* The Annals of Applied Statistics, 9(3), 1103–1140.

---

## 📖 Theoretical Overview

SLOPE is a regularized linear regression method designed to solve the problem of variable selection while controlling the **False Discovery Rate (FDR)**. The SLOPE estimator is defined as:

$$\hat{\beta} = \arg\min_{b \in \mathbb{R}^p} \frac{1}{2} \|y - Xb\|_2^2 + \sum_{i=1}^p \lambda_i |b|_{(i)}$$

where:
- $|b|_{(1)} \ge |b|_{(2)} \ge \dots \ge |b|_{(p)}$ are the sorted absolute values of the coefficients in decreasing order.
- $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_p \ge 0$ is a non-increasing sequence of regularization parameters.

### Penalty Sequences
1. **Benjamini-Hochberg Sequence ($\lambda_{BH}$):** Used for orthogonal designs:
   $$\lambda_{BH}(i) = \sigma \Phi^{-1}\left(1 - \frac{i \cdot q}{2p}\right)$$
   where $q$ is the target FDR level and $\Phi^{-1}$ is the standard normal quantile function.
2. **Gaussian-Adjusted Sequence ($\lambda_{G}^*$):** Corrects for correlation in Gaussian design matrices by recursively adjusting the variance using Wishart degrees of freedom:
   $$\lambda_G(i) = \lambda_{BH}(i) \sqrt{1 + \frac{1}{n - i} \sum_{j < i} \lambda_G(j)^2}$$
   The final sequence is flattened starting at the global minimum $k^*$ to maintain convexity: $\lambda_G^*(i) = \lambda_G(\min(i, k^*))$.

---

## ⚡ Algorithms Implemented

- **FastProxSL1 (Algorithm 4):** A stack-based pool-adjacent-violators-type algorithm that computes the proximal operator of the sorted $\ell_1$ norm in $O(p)$ time (after $O(p \log p)$ sorting).
- **FISTA Solver (Algorithm 2):** An accelerated proximal gradient method that fits SLOPE coefficients efficiently.
- **Scaled SLOPE (Algorithm 5):** An iterative algorithm that estimates the noise standard deviation $\sigma$ and fits SLOPE dynamically when the noise level is unknown.

---

## 📁 Repository Structure

```
.
├── README.md               # This documentation
├── run_all.py              # Master script to run all simulations and analyses
├── slope/
│   ├── __init__.py         # Package exports
│   └── solvers.py          # Solvers, prox operator, and penalty sequences
├── simulations/
│   ├── __init__.py
│   ├── simulation_1.py     # Reproduces Section 1.3.3 metrics (FDR, Power, MSE)
│   └── simulation_2.py     # Reproduces Section 3.1 equicorrelated testing
├── real_data/
│   ├── __init__.py
│   └── real_data_analysis.py # Diabetes pairwise interactions benchmark
├── tests/
│   └── test_slope.py       # Unit tests verifying solvers and prox operator
├── paper/
│   └── Bogdan et al. - 2015 - Slope—adaptive variable selection via convex optimization.pdf
├── figures/                # Output plots
└── tables/                 # Summary datasets
```

---

## 🚀 Getting Started

### Prerequisites
Make sure you have Python 3.9+ and the package manager `uv` installed. If you don't have `uv`, install it via pip:
```bash
pip install uv
```

### Installation
Clone this repository and run a test to ensure everything is set up:
```bash
git clone https://github.com/rkmishra1/SLOPE.git
cd SLOPE
uv run --with numpy --with scipy python -m unittest discover -s tests
```

### Run All Experiments
To run the full suite of simulations and the real data analysis:

```bash
# Run in fast mode (fewer replicates, ~10s execution)
uv run --with numpy --with scipy --with matplotlib --with scikit-learn --with pandas python run_all.py --mode fast

# Run in full mode (more replicates, ~5 mins execution)
uv run --with numpy --with scipy --with matplotlib --with scikit-learn --with pandas python run_all.py --mode full
```

---

## 📊 Summary of Results

### Simulation 1: Metrics comparison
Controls FDR at $q=0.1$ while achieving high power and lowest relative MSE:
- **SLOPE** (de-biased): Controls FDR, has high power, and yields the best prediction performance.
- **Lasso-Bonferroni**: Overly conservative, has low power.
- **Lasso-CV**: High power, but suffers from an inflated FDR (~80%).

*Generated Plot:* `figures/sim1_metrics.png`

### Simulation 2: Equicorrelated noise multiple testing
- **Marginal BH** is conservative and exhibits high variance in False Discovery Proportion.
- **SLOPE** (whitened) uses the covariance structure to achieve significantly higher power and stable FDR control.

*Generated Plot:* `figures/sim2_correlated.png`

### Real Data: Diabetes pairwise interactions ($n=442, p=55$)
Sparsity and fit comparison of SLOPE versus competing methods (saved in `tables/real_data_summary.csv`):

| Method | Selected Variables | R-squared ($R^2$) | Noise $\sigma$ Estimate |
| :--- | :---: | :---: | :---: |
| **SLOPE** | 41 | 0.5761 | 52.7029 |
| **Lasso-Bonf** | 37 | 0.5748 | 53.1148 |
| **Lasso-CV** | 14 | 0.5301 | *N/A* |
| **Stepwise-BIC** | 7 | 0.5340 | *N/A* |
| **Marginal-BH** | 14 | 0.5183 | *N/A* |
| **Full-model-BH** | 3 | 0.3998 | 53.1148 |

*Generated Plot:* `figures/real_data_coefs.png`
