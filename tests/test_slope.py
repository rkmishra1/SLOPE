import unittest
import numpy as np
from scipy.optimize import minimize
from slope.solvers import fast_prox_sl1, prox_sorted_l1, fista_slope, lambda_bh, lambda_g_star

class TestSLOPE(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

    def test_fast_prox_sl1_simple(self):
        # Basic test of FastProxSL1 monotonicity restoration
        y = np.array([3.0, 2.0, 4.0]) # Unsorted input to FastProxSL1 violates monotonicity
        lmbda = np.array([0.5, 0.3, 0.1])
        # After y - lmbda: [2.5, 1.7, 3.9]
        # Restoring monotonicity by averaging should yield a flat [2.7, 2.7, 2.7]
        x = fast_prox_sl1(y, lmbda)
        expected = np.array([2.7, 2.7, 2.7])
        np.testing.assert_allclose(x, expected, rtol=1e-5)

    def test_prox_sorted_l1_numeric(self):
        # Verify the proximal operator mathematically against scipy minimize
        v = np.array([-4.0, 1.5, -2.5])
        lmbda = np.array([1.5, 1.0, 0.5])
        
        # Calculate analytically using our FastProxSL1 implementation
        x_ana = prox_sorted_l1(v, lmbda)
        
        # Define objective for numerical minimization
        def obj(x):
            # 0.5 * ||x - v||_2^2 + J_lambda(x)
            diff = 0.5 * np.sum((x - v)**2)
            sorted_abs_x = np.sort(np.abs(x))[::-1]
            penalty = np.sum(lmbda * sorted_abs_x)
            return diff + penalty
            
        res = minimize(obj, x0=np.zeros_like(v), method='BFGS')
        x_num = res.x
        
        np.testing.assert_allclose(x_ana, x_num, atol=1e-4)

    def test_fista_slope_convergence(self):
        # Small-scale regression test
        n, p = 100, 10
        X = np.random.randn(n, p)
        # Standardize columns
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        
        beta_true = np.zeros(p)
        beta_true[0] = 3.0
        beta_true[1] = -2.0
        
        y = X.dot(beta_true) + 0.5 * np.random.randn(n)
        
        # Target FDR q = 0.05
        lmbda = lambda_bh(p, q=0.05, sigma=0.5)
        
        beta_est = fista_slope(X, y, lmbda)
        
        # Check that it identifies the correct support
        active_indices = np.where(np.abs(beta_est) > 1e-2)[0]
        self.assertIn(0, active_indices)
        self.assertIn(1, active_indices)
        
        # Check that inactive variables are set to 0 (or close to 0)
        for i in range(2, p):
            self.assertLess(np.abs(beta_est[i]), 0.1)

    def test_lambda_sequences(self):
        p = 50
        n = 100
        l_bh = lambda_bh(p, q=0.1)
        l_g = lambda_g_star(n, p, q=0.1)
        
        # Check properties
        self.assertEqual(len(l_bh), p)
        self.assertEqual(len(l_g), p)
        
        # Both must be non-negative and non-increasing
        self.assertTrue(np.all(l_bh >= 0))
        self.assertTrue(np.all(l_g >= 0))
        self.assertTrue(np.all(np.diff(l_bh) <= 1e-8))
        self.assertTrue(np.all(np.diff(l_g) <= 1e-8))
        
        # lambda_G* should be larger than lambda_BH (more conservative)
        self.assertTrue(np.all(l_g >= l_bh - 1e-8))

if __name__ == '__main__':
    unittest.main()
