import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm


def fast_prox_sl1(y: NDArray, lmbda: NDArray) -> NDArray:
    """
    FastProxSL1 via pool-adjacent-violators (Algorithm 4, Bogdan et al. 2015).

    Parameters
    ----------
    y : non-negative, non-increasing 1-D array
    lmbda : non-negative, non-increasing 1-D array, same length as y

    Returns
    -------
    x : non-negative, non-increasing 1-D array
    """
    n = len(y)
    # Each entry: [start, end, sum(y-lmbda), thresholded_avg]
    stack: list[list] = []

    for k in range(n):
        val = y[k] - lmbda[k]
        stack.append([k, k, val, max(0.0, val)])

        # merge while monotonicity is violated: w_{t-1} <= w_t
        while len(stack) > 1 and stack[-2][3] <= stack[-1][3]:
            b = stack.pop()
            p = stack[-1]
            p[1] = b[1]
            p[2] += b[2]
            p[3] = max(0.0, p[2] / (p[1] - p[0] + 1))

    x = np.zeros(n)
    for b in stack:
        x[b[0] : b[1] + 1] = b[3]
    return x


def prox_sorted_l1(v: ArrayLike, lmbda: ArrayLike) -> NDArray:
    """
    Proximal operator of the sorted L1 norm.

    Solves: argmin_x  0.5 ||v - x||^2 + sum_i lmbda_i |x|_(i)
    """
    v = np.asarray(v, dtype=float)
    lmbda = np.asarray(lmbda, dtype=float)

    s = np.sign(v)
    y = np.abs(v)
    sort_idx = np.argsort(y)[::-1]
    x_sorted = fast_prox_sl1(y[sort_idx], lmbda)
    return x_sorted[np.argsort(sort_idx)] * s


def fista_slope(
    X: ArrayLike,
    y: ArrayLike,
    lmbda: ArrayLike,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> NDArray:
    """
    FISTA solver for SLOPE (Algorithm 2, Bogdan et al. 2015).

    Parameters
    ----------
    X : (n, p) design matrix
    y : (n,) response vector
    lmbda : (p,) non-increasing penalty sequence
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    lmbda = np.asarray(lmbda, dtype=float)

    p = X.shape[1]
    L = np.linalg.norm(X, ord=2) ** 2  # Lipschitz constant

    beta = np.zeros(p)
    a = np.zeros(p)
    t = 1.0

    for _ in range(max_iter):
        beta_new = prox_sorted_l1(a - X.T @ (X @ a - y) / L, lmbda / L)

        if np.linalg.norm(beta_new - beta, ord=np.inf) < tol:
            return beta_new

        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t**2))
        a = beta_new + ((t - 1.0) / t_new) * (beta_new - beta)
        beta, t = beta_new, t_new

    return beta


def scaled_slope(
    X: ArrayLike,
    y: ArrayLike,
    q: float,
    design_type: str = "gaussian",
    max_iter: int = 100,
    tol: float = 1e-4,
) -> tuple[NDArray, float]:
    """
    Iterative SLOPE with unknown sigma (Algorithm 5, Bogdan et al. 2015).

    Returns
    -------
    beta : coefficient vector
    sigma_est : estimated noise standard deviation
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape

    l_s = lambda_bh(p, q) if design_type == "orthogonal" else lambda_g_star(n, p, q)

    S: set[int] = set()
    beta = np.zeros(p)
    sigma_est = float(np.std(y))

    for _ in range(max_iter):
        if len(S) == 0:
            rss, df = float(np.sum(y**2)), n - 1
        else:
            S_list = list(S)
            X_S = X[:, S_list]
            resid = y - X_S @ (np.linalg.pinv(X_S) @ y)
            rss, df = float(np.sum(resid**2)), n - len(S) - 1

        sigma_new = float(np.sqrt(rss / max(1.0, df)))
        beta_new = fista_slope(X, y, sigma_new * l_s)
        S_new = set(int(i) for i in np.where(np.abs(beta_new) > 1e-5)[0])

        if S_new == S and abs(sigma_new - sigma_est) < tol:
            return beta_new, sigma_new

        S, beta, sigma_est = S_new, beta_new, sigma_new

    return beta, sigma_est


def lambda_bh(p: int, q: float, sigma: float = 1.0) -> NDArray:
    """Benjamini-Hochberg penalty sequence: lambda_i = sigma * Phi^{-1}(1 - iq / 2p)."""
    i = np.arange(1, p + 1)
    return sigma * norm.ppf(1.0 - i * q / (2.0 * p))


def lambda_g_star(n: int, p: int, q: float, sigma: float = 1.0) -> NDArray:
    """
    Gaussian-adjusted penalty sequence lambda_G* (Section 3.2.2, Bogdan et al. 2015).

    Recursively inflates lambda_BH by the Wishart correction, then flattens
    from the global minimum k* onward to preserve convexity.
    """
    l_bh = lambda_bh(p, q, sigma=sigma)
    l_g = np.zeros(p)
    l_g[0] = l_bh[0]

    limit = min(p, n - 1)
    sum_sq = l_g[0] ** 2
    for i in range(1, limit):
        l_g[i] = l_bh[i] * np.sqrt(1.0 + sum_sq / (n - i - 1))
        sum_sq += l_g[i] ** 2

    if p >= n:
        l_g[limit:] = np.inf

    k_star = int(np.argmin(l_g[: limit + 1] if p < n else l_g[:limit]))
    l_g_star = l_g.copy()
    l_g_star[k_star:] = l_g[k_star]
    if p >= n:
        l_g_star[limit:] = l_g[k_star]

    return l_g_star


def lambda_mc(
    X: ArrayLike,
    q: float,
    sigma: float = 1.0,
    num_draws: int = 500,
    max_k: int = 100,
) -> NDArray:
    """
    Monte Carlo adjusted penalty sequence lambda_MC (Section 3.2.2, Bogdan et al. 2015).
    """
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    l_bh = lambda_bh(p, q, sigma=sigma)
    l_mc = np.zeros(p)
    l_mc[0] = l_bh[0]

    limit = min(p, n - 1, max_k)
    for i in range(1, limit):
        u_sq_sum = 0.0
        for _ in range(num_draws):
            S_idx = np.random.choice(p, size=i, replace=False)
            S_set = set(S_idx.tolist())
            j = int(np.random.choice([x for x in range(p) if x not in S_set]))

            X_S, X_j = X[:, S_idx], X[:, j]
            A = X_S.T @ X_S
            try:
                h = np.linalg.solve(A, l_mc[:i])
            except np.linalg.LinAlgError:
                h = np.linalg.pinv(A) @ l_mc[:i]
            u_sq_sum += float(X_j @ (X_S @ h)) ** 2

        l_mc[i] = l_bh[i] * np.sqrt(1.0 + u_sq_sum / num_draws)

    k_star = int(np.argmin(l_mc[:limit]))
    l_mc[k_star:] = l_mc[k_star]
    return l_mc
