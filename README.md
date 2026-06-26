# SLOPE: Sorted L-One Penalized Estimation

Python reference implementation of **SLOPE** — reproducing the simulations and benchmark from:

> Bogdan, M., van den Berg, E., Sabatti, C., Su, W., & Candès, E. J. (2015). *SLOPE—Adaptive Variable Selection via Convex Optimization.* The Annals of Applied Statistics, 9(3), 1103–1140.

---

## Theory

SLOPE minimizes a sorted-L1-penalized least squares objective:

$$\hat{\beta} = \arg\min_{b \in \mathbb{R}^p} \frac{1}{2} \|y - Xb\|_2^2 + \sum_{i=1}^p \lambda_i |b|_{(i)}$$

where $|b|_{(1)} \ge \cdots \ge |b|_{(p)}$ are the sorted absolute coefficients and $\lambda_1 \ge \cdots \ge \lambda_p \ge 0$.

### Penalty sequences

| Sequence | Formula | When to use |
|---|---|---|
| **BH** ($\lambda_{BH}$) | $\sigma\,\Phi^{-1}\!\left(1 - \tfrac{iq}{2p}\right)$ | Orthogonal designs |
| **Gaussian** ($\lambda_G^*$) | Wishart-corrected BH, flattened at global min $k^*$ | Gaussian / correlated designs |

The Gaussian sequence corrects for correlation recursively:
$$\lambda_G(i) = \lambda_{BH}(i)\sqrt{1 + \frac{1}{n-i}\sum_{j<i}\lambda_G(j)^2}$$
then flattens: $\lambda_G^*(i) = \lambda_G(\min(i, k^*))$.

---

## Algorithms

| Function | Paper | Description |
|---|---|---|
| `fast_prox_sl1` | Alg. 4 | PAV-style proximal operator for sorted-L1, $O(p)$ after sort |
| `fista_slope` | Alg. 2 | Accelerated proximal gradient (FISTA) |
| `scaled_slope` | Alg. 5 | Iterative SLOPE when $\sigma$ is unknown |

---

## Repository structure

```
slope/solvers.py          # All algorithms and penalty sequences
simulations/
  simulation_1.py         # Section 1.3.3 — FDR / Power / MSE
  simulation_2.py         # Section 3.1  — equicorrelated noise
real_data/
  real_data_analysis.py   # Diabetes interactions benchmark (n=442, p=55)
tests/test_slope.py       # Unit tests
run_all.py                # Run everything
```

---

## Getting started

**Requirements:** Python 3.9+, [`uv`](https://github.com/astral-sh/uv)

```bash
git clone https://github.com/rkmishra1/SLOPE.git
cd SLOPE
pip install -e .                                        # install package
uv run --with numpy --with scipy python -m unittest discover -s tests
```

**Run experiments:**

```bash
# fast (~10s, 5 replicates)
uv run --with numpy --with scipy --with matplotlib --with scikit-learn --with pandas \
    python run_all.py --mode fast

# full (~5 min, 100 replicates)
uv run --with numpy --with scipy --with matplotlib --with scikit-learn --with pandas \
    python run_all.py --mode full
```

---

## Results

### Simulation 1 — FDR / Power / MSE (`figures/sim1_metrics.png`)

At $q = 0.1$: SLOPE (de-biased) controls FDR and achieves the lowest relative MSE. Lasso-Bonferroni is overly conservative; Lasso-CV inflates FDR to ~80%.

### Simulation 2 — Equicorrelated noise (`figures/sim2_correlated.png`)

Whitened SLOPE exploits the covariance structure for higher power and stable FDR. Marginal BH is conservative with high variance in false discovery proportion.

### Real data — Diabetes interactions (`tables/real_data_summary.csv`, `figures/real_data_coefs.png`)

| Method | Selected | $R^2$ | $\hat\sigma$ |
|:---|:---:|:---:|:---:|
| **SLOPE** | 41 | 0.5761 | 52.70 |
| Lasso-Bonf | 37 | 0.5748 | 53.11 |
| Lasso-CV | 14 | 0.5301 | — |
| Stepwise-BIC | 7 | 0.5340 | — |
| Marginal-BH | 14 | 0.5183 | — |
| Full-model-BH | 3 | 0.3998 | 53.11 |
