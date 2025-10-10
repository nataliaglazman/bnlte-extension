from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import scipy.stats as st
from scipy.interpolate import BSpline
from scipy.stats import geninvgauss
from functools import lru_cache
from numba import jit, prange
import numba as nb
from statsmodels import robust
from scipy.special import logsumexp
import math
from scipy.special import kv
from numba import njit, prange
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression




@jit(nopython=True, cache=True, parallel=True)
def compute_edge_contributions_parallel(
    X: np.ndarray,
    beta: np.ndarray,
    r: np.ndarray,
    basis_beta: np.ndarray,
    no_incoming: np.ndarray,
    forbidden: np.ndarray
) -> np.ndarray:
    """Parallel computation of edge contributions"""
    n, p = X.shape
    edge_contrib = np.zeros((n, p))
    
    for j in prange(p):
        if no_incoming[j]:
            continue
        col = np.zeros(n)
        for l in range(p):
            if l == j or forbidden[j, l]:
                continue
            coeff = beta[j, l] * r[j, l]
            if coeff.any():
                col += X[:, l] * (basis_beta @ coeff)
        edge_contrib[:, j] = col
    
    return edge_contrib

@jit(nopython=True, cache=True)
def has_cycle_fast(A: np.ndarray) -> bool:
    """Fast cycle detection using topological sort"""
    indeg = A.sum(0)
    stack = []
    p = A.shape[0]
    
    for v in range(p):
        if indeg[v] == 0:
            stack.append(v)
    
    seen = 0
    while len(stack) > 0:
        v = stack.pop()
        seen += 1
        for w in range(p):
            if A[v, w] > 0:
                indeg[w] -= 1
                if indeg[w] == 0:
                    stack.append(w)


    
    return seen < p

@jit(nopython=True, cache=True)
def compute_residuals_fast(X: np.ndarray, baseline: np.ndarray, edges: np.ndarray, no_incoming: np.ndarray) -> np.ndarray:
    residuals = X - baseline - edges
    for j in range(X.shape[1]):
        if no_incoming[j]:
            residuals[:, j] = 0.0  
    return residuals


def gig_rvs(a: float, b: float, p: float, rng=np.random.default_rng()):
    """Original GIG RVS for compatibility"""
    chi = b
    psi = a
    omega = np.sqrt(chi * psi)
    tau = geninvgauss.rvs(p=p, b=omega, random_state=rng)
    return tau


@lru_cache(maxsize=32)
def bspline_basis_cached(n: int, K: int, degree: int = 3) -> tuple:
    """Cache B-spline basis computation for common sizes"""
    z = np.linspace(0.1, 0.9, n)
    return bspline_basis(z, K=K, degree=degree), z


def bspline_basis(z: np.ndarray, *, K: int, degree: int = 3) -> np.ndarray:
    """Return an (n×K) open‑uniform B‑spline design matrix."""
    z = np.asarray(z, float)
    if z.ndim != 1:
        raise ValueError("z must be 1‑D")
    if K <= degree:
        raise ValueError("need at least degree+1 basis functions")

    n_internal = K - degree - 1
    internal = np.linspace(0, 1, n_internal + 2)[1:-1]
    t = np.r_[np.zeros(degree + 1), internal, np.ones(degree + 1)]

    # Vectorized basis computation
    basis = np.empty((len(z), K))
    for k in range(K):
        coeff = np.zeros(K)
        coeff[k] = 1.0
        basis[:, k] = BSpline(t, coeff, degree)(z)
    return basis


def bspline_basis_single_row(z_val: float, *, K: int, degree: int = 3, t: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute a single row of the B‑spline basis matrix (for speed)."""
    if t is None:
        n_internal = K - degree - 1
        internal = np.linspace(0, 1, n_internal + 2)[1:-1]
        t = np.r_[np.zeros(degree + 1), internal, np.ones(degree + 1)]

    basis_row = np.empty(K)
    for k in range(K):
        coeff = np.zeros(K)
        coeff[k] = 1.0
        basis_row[k] = BSpline(t, coeff, degree)(z_val)
    return basis_row


def _second_diff_penalty(K: int) -> np.ndarray:
    R = np.zeros((K - 2, K))
    for k in range(K - 2):
        R[k, k] = 1.0
        R[k, k + 1] = -2.0
        R[k, k + 2] = 1.0
    return R.T @ R



def tensor_product_basis_row(v: np.ndarray, *, K: int, degree: int = 3, t: np.ndarray = None) -> np.ndarray:
    """Compute the tensor product B-spline basis row for a multivariate point v (shape (d,))."""
    d = len(v)
    if d != 5:
        raise ValueError("Currently supports exactly 5 variables.")
    basis_list = [bspline_basis_single_row(v[j], K=K, degree=degree, t=t) for j in range(d)]
    # Use einsum for efficient multi-dimensional outer product
    return np.einsum('i,j,k,l,m -> ijklm', *basis_list).flatten()


def bspline_basis_derivative(z: float, K: int, degree: int, t: np.ndarray) -> np.ndarray:
    """Compute the derivative of B-spline basis functions with respect to z.
    
    Args:
        z (float): Input value, typically in [0, 1].
        K (int): Number of basis functions.
        degree (int): Degree of the B-spline (e.g., 3 for cubic).
        t (np.ndarray): Knot vector, shape (K + degree + 1,).
    
    Returns:
        np.ndarray: Derivative of B-spline basis functions, shape (K,).
    """
    if degree == 0:
        return np.zeros(K)  # Derivative of degree-0 B-spline is zero
    
    # Initialize derivative array
    db = np.zeros(K)
    
    # Compute basis functions for degree-1
    b_lower = np.zeros(K + 1)  # Need K+1 for degree-1 basis
    for i in range(K + 1):
        if i >= len(t) - degree or t[i + degree] == t[i]:
            continue
        # Compute B_{i,degree-1}(z)
        b_lower[i] = bspline_basis_single_row(z, K=K + 1, degree=degree - 1, t=t)[i]
    
    # Compute derivative using recursive formula
    for i in range(K):
        if t[i + degree] == t[i] or t[i + degree + 1] == t[i + 1]:
            db[i] = 0.0  # Handle zero denominators
        else:
            db[i] = degree * (
                b_lower[i] / (t[i + degree] - t[i]) -
                b_lower[i + 1] / (t[i + degree + 1] - t[i + 1])
            )
    
    return db

def _make_t(K: int, degree: int) -> np.ndarray:
    """Generate knot vector for B-spline basis."""
    n_knots = K + degree + 1
    knots = np.linspace(0, 1, n_knots - 2 * degree)
    knots = np.concatenate([np.zeros(degree), knots, np.ones(degree)])
    return knots

def additive_basis_row(v: np.ndarray, *, K: int, degree: int = 3, t: np.ndarray = None) -> np.ndarray:
    """Compute additive B-spline basis row for a multivariate point v (shape (d,))."""
    d = len(v)
    basis_list = [bspline_basis_single_row(v[j], K=K, degree=degree, t=t) for j in range(d)]
    return np.concatenate(basis_list)  # Shape: (d * K,)



@dataclass
class StateBS:
    sigma2: np.ndarray            # (p,)
    beta: np.ndarray              # (p×p×K_beta)
    gamma: np.ndarray             # (p×K_gamma)
    r: np.ndarray                 # (p×p×K_beta)
    z: np.ndarray                 # (n,)
    tau_gamma: np.ndarray         # (p,)
    sigma3: np.ndarray            # (p,) 
    rho: float                    # global edge inclusion probability
    C: np.ndarray                 # (3,3) for longitudinal data
    eta: np.ndarray               # (n,)

class BN_LTE_MCMC_BSpline_Optimized:

    class Hyper:
        def __init__(
            self,
            tau_beta: float = 15.0,
            tau_gamma: float | None = None,
            nu_sigma: float = 0.001,
            lambda_sigma: float = 0.001,
            a_s: float = 1.0,
            b_s: float = 0.1,
            pi_edge: float = 0.5,
            a_rho: float = 1.0,
            b_rho: float = 1.0,
            a_sigma: float = 5.0,
            b_sigma: float = 5.0,
            tau_C: float = 15.0,
            nu_eta: float = 0.001,
            lambda_eta: float = 0.001,
            tau_phi: float = 15.0,
        ) -> None:
            self.tau_beta = tau_beta
            self.tau_gamma = tau_beta if tau_gamma is None else tau_gamma
            self.pi_edge = pi_edge
            self.a_rho = a_rho
            self.b_rho = b_rho
            self.nu_sigma = nu_sigma
            self.lambda_sigma = lambda_sigma
            self.a_s = a_s
            self.b_s = b_s
            self.a_sigma = a_sigma
            self.b_sigma = b_sigma
            self.tau_C = tau_C
            self.nu_eta = nu_eta
            self.lambda_eta = lambda_eta
            self.tau_phi = tau_phi

    def __init__(
        self,
        X: np.ndarray,
        *,
        K_edge: int = 7,
        K_baseline: Optional[int] = 5,
        static_features: Optional[Sequence[int]] = None,
        forbidden_edges: Optional[list] = None,
        hypers: Optional["BN_LTE_MCMC_BSpline_Optimized.Hyper"] = None,
        seed: Optional[int] = None,
        print_every: int = 10,
        families: Optional[List[List[int]]] = None,
        family_matrix: Optional[np.ndarray] = None,
        degree: int = 3,
        scaler
    ) -> None:
        if seed is not None:
            np.random.seed(seed)

        self.K_beta = K_edge
        self.K_gamma = K_baseline if K_baseline is not None else K_edge
        self.K = self.K_beta
        self.a_t = 1.0
        self.b_t = 0.1
        self.seed = seed


        self.families = families if families is not None else []
        self.family_matrix = family_matrix
        
        self.X_time = X
        self.valid_time_mask = ~np.isnan(self.X_time).any(axis=2)  # Shape (T, n), True where data is valid
        print('Valid time mask shape:', self.valid_time_mask.shape)
        X = X[0]
        self.X = np.asarray(X, np.float64)  
        self.n, self.p = self.X.shape

        self.corr_matrix = np.corrcoef(self.X, rowvar=False)

        size_indices = [7,8,9,10]  # Placeholder; customize

        pca = PCA(n_components=1)
        self.pc1 = pca.fit_transform(self.X)[:, 0]
        self.pc_mean = np.mean(self.pc1)
        if len(size_indices) > 0:
            # Regress out size effects: fit linear models for each var on size vars
            size_X = self.X[:, size_indices]
            adjusted_X = self.X.copy()
            for j in range(self.p):
                if j not in size_indices:
                    reg = LinearRegression(fit_intercept=True).fit(size_X, self.X[:, j])
                    adjusted_X[:, j] -= reg.predict(size_X)
            # Recompute PCA on adjusted data
            pca = PCA(n_components=1)
            self.pc1 = pca.fit_transform(adjusted_X)[:, 0]
            self.pc_direction = np.sign(np.mean(pca.components_[0]))
            self.pc1 *= -self.pc_direction

        self.scaler = scaler

        self.original_vars = np.var(self.X * self.scaler.scale_ + self.scaler.mean_, axis=0)

        # self.pc_direction = np.sign(np.mean(pca.components_[0]))  # Optional: Adjust sign if you know the expected direction (e.g., positive if most biomarkers increase with progression)
        # print(f"PCA direction: {self.pc_direction}")
        # self.pc1 *= self.pc_direction  # Flip PC1 if needed to ensure "positive" direction aligns with expected progression

        self.h = hypers or BN_LTE_MCMC_BSpline_Optimized.Hyper()

        if static_features is None:
            static_features = []
        self.static_features = np.asarray(static_features, int)
        if self.static_features.size and (
            np.any(self.static_features < 0) or np.any(self.static_features >= self.p)
        ):
            raise ValueError("static_features indices out of range")
        
        self._no_incoming = np.zeros(self.p, bool)
        if self.static_features.size > 0:
            self._no_incoming[self.static_features] = True

        def _make_t(K: int, degree: int = 3):
            n_internal = K - degree - 1
            internal = np.linspace(0, 1, n_internal + 2)[1:-1]
            return np.r_[np.zeros(degree + 1), internal, np.ones(degree + 1)]

        self.degree = degree
        self.t_beta = _make_t(self.K_beta, self.degree)
        self.t_gamma = _make_t(self.K_gamma, self.degree)

        self.K_c = 6
        self.degree = degree

        sigma3 =  st.gamma.rvs(a=self.a_t, scale=self.b_t)
        self.var_indices = [0, 1, 2,3,4]  # Indices of the 3 variables to use for time modeling
        z0 = np.random.uniform(0.1, 0.9, self.n)
        # c0 = np.random.rand(3, self.K_beta)
        c0 = np.random.rand(len(self.var_indices) * self.K_c)

        self._basis_beta = bspline_basis(z0, K=self.K_beta)
        self._basis_gamma = bspline_basis(z0, K=self.K_gamma)

        self._BtB_beta = self._basis_beta.T @ self._basis_beta
        self._BtB_gamma = self._basis_gamma.T @ self._basis_gamma

        # State arrays
        beta0 = np.zeros((self.p, self.p, self.K_beta))
        gamma0 = np.zeros((self.p, self.K_gamma))
        r0 = np.zeros((self.p, self.p, self.K_beta))
        self.forbidden = np.zeros((self.p, self.p), dtype=bool)
        eta0 = np.random.uniform(0.001, 0.01, self.n)


        if forbidden_edges is not None:
            if isinstance(forbidden_edges, np.ndarray):
                if forbidden_edges.shape != (self.p, self.p):
                    raise ValueError("forbidden_edges mask must have shape (p,p)")
                self.forbidden |= forbidden_edges.astype(bool)
            else:
                for j, l in forbidden_edges:
                    self.forbidden[j, l] = True
        self.rng = np.random.default_rng(seed)


        sigma2_0 = st.invgauss.rvs(0.001, scale=0.001, size=self.p)
        tau0 = np.full(self.p, 1.0)
        tau0 = st.gamma.rvs(a=1, scale=0.1, size=self.p)
        rho = self.h.pi_edge
        self.state = StateBS(sigma2_0, beta0, gamma0, r0, z0, tau0, sigma3, rho, c0, eta0)
        
        self.alpha = 1.0

        self._basis = self._basis_beta  

        self._edge_contrib = np.zeros_like(self.X)
        self._baseline_contrib = np.zeros_like(self.X)
        

        self._update_all_edge_contrib()
        self._update_baseline_contrib()

        self._acc = dict(z_prop=0, z_acc=0)
        self.z_step = 0.2
        self.target_accept_rate = 0.25
        self.step_adapt_rate = 0.05

        self.print_every = print_every
        self.T = 3
        

        # Create knot vector for time B-splines
        n_internal = self.K_c - self.degree - 1
        internal = np.linspace(0, 1, n_internal + 2)[1:-1]
        self.t_time = np.r_[np.zeros(self.degree + 1), internal, np.ones(self.degree + 1)]

        
        
        V = self.X_time[:, :, self.var_indices]  # (T, n, 3)
        print('V shape:', V.shape)
        valid_V = V[self.valid_time_mask, :]
        self.v_min = valid_V.min(axis=0)  # (3,)
        self.v_max = valid_V.max(axis=0)
        self.v_max[self.v_max == self.v_min] += 1e-6
        self.V_norm = np.full_like(V, np.nan)  # (T, n, 3)
        print('V_norm shape before:', self.V_norm.shape)
        self.V_norm[self.valid_time_mask, :] = (V[self.valid_time_mask, :] - self.v_min) / (self.v_max - self.v_min)
        print('V_norm shape:', self.V_norm.shape)

        self.t_delta = _make_t(self.K_beta, self.degree)
        # self._basis_C = np.array([tensor_product_basis_row(self.V_norm[0,i, :], K=self.K_c, degree=3, t=self.t_beta)
        #                       for i in range(self.n)])
        self._basis_C = np.array([additive_basis_row(self.V_norm[0,i,:], K=self.K_c, degree=3, t=self.t_time)
                          for i in range(self.n)])  # Shape: (n, d * K_c)

        print("Basis C shape:", self._basis_C.shape)  # Should be (n, K_c^3)


        self.RtR_gamma = _second_diff_penalty(self.K_gamma)
        

        self._chol_cache = {}




    def _update_baseline_contrib(self) -> None:
        """Vectorized baseline contribution update"""
        self._baseline_contrib = self._basis_gamma @ self.state.gamma.T

    def _update_edge_contrib(self, column: int):
        """Update edge contribution for a single column"""
        j = column
        col = np.zeros(self.n)
        if not self._no_incoming[j]:
            for l in range(self.p):
                if l != j and not self.forbidden[j, l]:
                    coeff = self.state.beta[j, l] * self.state.r[j, l]
                    if coeff.any():
                        col += self.X[:, l] * (self._basis_beta @ coeff)
        self._edge_contrib[:, j] = col

    def _update_all_edge_contrib(self):
        """Update all edge contributions at once"""
        for j in range(self.p):
            self._update_edge_contrib(j)

    def _B_from_basis_row(self, basis_row_beta: np.ndarray) -> np.ndarray:
        """Optimized B matrix computation"""
        B = np.zeros((self.p, self.p))
        active_mask = ~self._no_incoming
        
        for j in np.where(active_mask)[0]:
            for l in range(self.p):
                if l != j and not self.forbidden[j, l]:
                    coeff = self.state.beta[j, l] * self.state.r[j, l]
                    if coeff.any():
                        B[j, l] = basis_row_beta @ coeff
        return B
    
    def _update_sigma2(self) -> None:
        """Vectorized sigma2 update"""
        resid = compute_residuals_fast(self.X, self._baseline_contrib, self._edge_contrib, self._no_incoming)
        
        shape = self.h.nu_sigma + (self.n * 0.5)
        
        
        for j in range(self.p):
            if self._no_incoming[j]:
                self.state.sigma2[j] = 1
            else:
                scale = self.h.lambda_sigma + 0.5 * np.sum(resid[:, j] ** 2)
                sigma = st.invgamma.rvs(a=shape, scale=scale)
                if sigma > 1e6:
                    print(f"Residuals: {resid[:, j]}")

                self.state.sigma2[j] = np.clip(sigma, 1e-6, 1e6)



    def _update_gamma(self) -> None:
        """Optimized gamma update with cached matrices and diagnostics"""
        sigma2 = self.state.sigma2
        BtB = self._BtB_gamma
        
        for j in range(self.p):
            if self._no_incoming[j]:
                self.state.gamma[j, :] = 1e-8
                self._baseline_contrib[:, j] = self.X[:, j]
                continue
            
            y = self.X[:, j] - self._edge_contrib[:, j]
            tau_gamma = np.clip(self.state.tau_gamma[j], 1e-2, 1e4)  # Bound tau_gamma
            var_scale = self.original_vars[j] / np.median(self.original_vars)  # Relative to median (agnostic)
            penalty_scale = 5.0 + np.log1p(var_scale)  # Mild boost for high-var (e.g., +1–3x)
            # print(f"Var scale for var {j}: {var_scale}, penalty scale: {penalty_scale}")
            # Qt = BtB / sigma2[j] + (self.RtR_gamma * penalty_scale) / tau_gamma + 1e-4 * np.eye(self.K_gamma)
            Qt = BtB / sigma2[j] + self.RtR_gamma / (tau_gamma) + 1e-4 * np.eye(self.K_gamma)
            
            # Log diagnostics
            cond_num = np.linalg.cond(Qt)
            if cond_num > 1e6:
                print(f"Warning: High condition number for Qt (j={j}): {cond_num}")
            
            try:
                L = np.linalg.cholesky(Qt)
                # m = np.linalg.solve(L.T, np.linalg.solve(L, self._basis_gamma.T @ y)) / sigma2[j]
                m = np.linalg.solve(L.T, np.linalg.solve(L, (self._basis_gamma.T @ y) / sigma2[j]))
                self.state.gamma[j] = st.multivariate_normal.rvs(mean=m, cov=np.linalg.inv(Qt))
            except np.linalg.LinAlgError:
                print(f"LinAlgError in _update_gamma, j={j}, sigma2={sigma2[j]}, tau_gamma={tau_gamma}")
                continue
        
        self._update_baseline_contrib()
            

    def _update_tau2_mh(self) -> None:
        """Metropolis-Hastings update for global sigma_t^2 as per paper"""
        rng = self.rng
        current = self.state.sigma3
        proposed = st.gamma.rvs(a=self.a_t, scale=self.b_t)
        log_alpha = 0.0
        for j in range(self.p):
            if self._no_incoming[j]:
                continue
            # Collect active terms
            active_mask = np.zeros((self.p, self.K_beta), bool)
            for l in range(self.p):
                if l == j or self.forbidden[j, l]:
                    continue
                active_mask[l] = self.state.r[j, l].astype(bool)
            m = active_mask.sum()
            if m == 0:
                continue
            # Build D (n x m)
            D = np.zeros((self.n, m))
            col = 0
            for l in range(self.p):
                for k in range(self.K_beta):
                    if active_mask[l, k]:
                        D[:, col] = self.X[:, l] * self._basis_beta[:, k]
                        col += 1
            A = D.T @ D  # m x m
            s2 = self.state.sigma2[j]
            resid = self.X[:, j] - self._baseline_contrib[:, j]
            res2 = np.sum(resid**2)
            dr = D.T @ resid  # m
            def compute_ll(sig_t2):
                ratio = sig_t2 / s2
                logdet = self.n * np.log(s2) + np.linalg.slogdet(np.eye(m) + ratio * A)[1]
                Q = np.eye(m) / sig_t2 + A / s2
                try:
                    L = np.linalg.cholesky(Q)
                    inner = np.linalg.solve(L.T, np.linalg.solve(L, dr / s2))
                    quad = res2 / s2 - np.dot(dr, inner) / s2
                except np.linalg.LinAlgError:
                    return -np.inf
                return -0.5 * logdet - 0.5 * quad
            ll_prop = compute_ll(proposed)
            ll_curr = compute_ll(current)
            log_alpha += ll_prop - ll_curr
        accept_prob = min(1.0, np.exp(np.clip(log_alpha, -50, 50)))
        if rng.random() < accept_prob:
            self.state.sigma3 = proposed


    def _update_beta_and_r_blocked(self) -> None:
        """Optimized beta and r update"""
        sigma2 = self.state.sigma2
        sigma3 = self.state.sigma3
        BtB = self._BtB_beta
        rng = self.rng

        self._r_prev = self.state.r.copy()

        X_products = {}
        for l in range(self.p):
            X_products[l] = self.X[:, l][:, None] * self._basis_beta


        for j in rng.permutation(self.p):
            if self._no_incoming[j]:
                continue
            for l in rng.permutation(self.p):
                if l == j or self.forbidden[j, l]:
                    continue
                r_temp = self.state.r[j, l].copy()
                for k in rng.permutation(self.K_beta):
                    self._update_edge_contrib(column=j)
                    y = self.X[:, j] - self._baseline_contrib[:, j] - self._edge_contrib[:, j]
                    D_full = X_products[l]
                    sig2 = sigma2[j]
                    sig3 = sigma3
                    d = D_full[:, k]
                    if r_temp[k]:
                        y += d * self.state.beta[j, l, k]
                    Qt = d.dot(d) / sig2 + 1.0 / sig3
                    m = d.dot(y) / sig2 / Qt
                    penalty_on = self._compute_family_penalty_fast(j, l, k, True)
                    penalty_off = self._compute_family_penalty_fast(j, l, k, False)
                    delta_penalty = penalty_on - penalty_off
                    ll_on = -0.5 * np.sum((y - d * m) ** 2) / sig2 - 0.5 * np.log(Qt)
                    ll_off = -0.5 * np.sum(y ** 2) / sig2
                    logit = (ll_on - ll_off) +np.log(self.state.rho) - np.log1p(-self.state.rho) + delta_penalty
 
                    p_on = 1.0 / (1.0 + np.exp(-np.clip(logit, -50, 50)))
                    r_temp[k] = rng.random() < p_on
                if r_temp.any() and not self.state.r[j, l].any():
                    edge_mat = self.state.r.any(axis=2).copy().T
                    edge_mat[l, j] = 1
                    if has_cycle_fast(edge_mat):
                        r_temp[:] = self.state.r[j, l]
                self.state.r[j, l] = r_temp
                            

        self._update_all_edge_contrib()

    def _update_beta(self) -> None:
        X_products = {}
        for l in range(self.p):

            X_products[l] = self.X[:, l][:, None] * self._basis_beta
        for j in range(self.p):
            
            if self._no_incoming[j]:
                continue
            
            y_base = self.X[:, j] - self._baseline_contrib[:, j]
            y = y_base.copy()
            
            for l in range(self.p):
                if l != j and not self.forbidden[j, l]:
                    mask = self.state.r[j, l].astype(bool)
                    if np.any(mask):
                        D = X_products[l]
                        y -= D[:, mask] @ self.state.beta[j, l, mask]
            
            sig2 = self.state.sigma2[j]
            
            for l in range(self.p):
                if l != j and not self.forbidden[j, l]:
                    mask = self.state.r[j, l].astype(bool)
                    if not np.any(mask):
                        self.state.beta[j, l] = 0.0
                        continue
                    
                    
                    D_act = X_products[l][:, mask]
                    Qt = D_act.T @ D_act / sig2 + np.eye(mask.sum()) / ((self.state.sigma3))
                    
                    try:
                        L = np.linalg.cholesky(Qt)
                        m = np.linalg.solve(L.T, np.linalg.solve(L, D_act.T @ y)) / sig2
                        beta_draw = st.multivariate_normal.rvs(mean=m, cov=np.linalg.inv(Qt), random_state=self.rng)
                        beta_draw = np.atleast_1d(beta_draw)
                        self.state.beta[j, l, :] = 0.0
                        self.state.beta[j, l, mask] = beta_draw
                        y -= D_act @ beta_draw
                    except np.linalg.LinAlgError:
                        continue

        self._update_all_edge_contrib()

    def _compute_family_penalty_fast(self, j: int, l: int, k: int, value: bool) -> float:
        """Fast family penalty computation for a single edge, only when both nodes are in the same family."""
        if self.family_matrix is None or not value:
            return 0.0
        
        penalty = 0.0
        lambda_penalty = 7.0
        
        # Only compute penalty if j and l are in the same family
        if self.family_matrix[j, l]:  # Check if j and l are in the same family
            family_members = np.where(self.family_matrix[l])[0]
            for member in family_members:
                if member != l and self.state.r[j, member].any():
                    rho = abs(self.corr_matrix[l, member])
                    penalty += rho
        
        return -lambda_penalty * penalty

    def _log_prior_z_delta(self, i: int, z_old: float, z_new: float) -> float:
        """Fast z prior delta computation"""
        a, b = 2.0, 2.0
        gamma = 0.1
        eps = 1e-9
        
        dbeta = (a - 1) * (np.log(z_new) - np.log(z_old)) + (b - 1) * (np.log1p(-z_new) - np.log1p(-z_old))
        
        z_others = np.delete(self.state.z, i)
        s_old = np.clip(np.sin(np.pi * np.abs(z_others - z_old)), eps, None)
        s_new = np.clip(np.sin(np.pi * np.abs(z_others - z_new)), eps, None)
        dcoul = (gamma / 2.0) * np.sum(np.log(s_new) - np.log(s_old))
        lambda_align = 0.5  # Adjust based on how strongly you want to enforce direction
        z_mean = np.mean(self.state.z)  # Approximate; full mean_z_new would require adjustment but is negligible for large n
        dalign = lambda_align * ((z_new - z_old) * (self.pc1[i] - self.pc_mean))

      
        return 0

    def _update_z_single_row(self, it) -> None:
        """Optimized z update"""
        sigma2 = self.state.sigma2
        gammaT = self.state.gamma.T
        inv_sigma2 = 1.0 / sigma2
        
        for i in range(self.n):
            self._acc["z_prop"] += 1
            z_old = self.state.z[i]
            z_prop = (z_old + np.random.normal(scale=self.z_step)) % 2.0
            if z_prop > 1:
                z_prop = 2 - z_prop

            b_old_beta = self._basis_beta[i]
            b_new_beta = bspline_basis_single_row(z_prop, K=self.K_beta, t=self.t_beta)
            b_old_gamma = self._basis_gamma[i]
            b_new_gamma = bspline_basis_single_row(z_prop, K=self.K_gamma, t=self.t_gamma)

            delta_edge = np.zeros(self.p)
            active_j = np.where(~self._no_incoming)[0]
            
            for j in active_j:
                edge_sum = 0.0
                for l in range(self.p):
                    if l != j and not self.forbidden[j, l]:
                        coeff = self.state.beta[j, l] * self.state.r[j, l]
                        v = b_new_beta @ coeff
                        if coeff.any():
                            edge_sum += self.X[i, l] * ((b_new_beta @ coeff) - (b_old_beta @ coeff))
                delta_edge[j] = edge_sum

            delta_baseline = (b_new_gamma - b_old_gamma) @ gammaT
            resid_old = self.X[i] - self._baseline_contrib[i] - self._edge_contrib[i]
            resid_new = resid_old - delta_baseline - delta_edge

            z_prop_vec = np.delete(self.state.z, i)
            z_prop_vec = np.insert(z_prop_vec, i, z_prop)

            lp_delta = self._log_prior_z_delta(i, z_old, z_prop)
            
            B_old = self._B_from_basis_row(b_old_beta)
            B_new = self._B_from_basis_row(b_new_beta)
            _, logdet_old = np.linalg.slogdet(np.eye(self.p) - B_old)
            _, logdet_new = np.linalg.slogdet(np.eye(self.p) - B_new)
            
            ll_delta_det = logdet_new - logdet_old
            active_j = np.where(~self._no_incoming)[0] 
            ll_delta = 0.5 * np.sum((resid_old[active_j] ** 2 - resid_new[active_j] ** 2) * inv_sigma2[active_j])

            if np.log(np.random.rand()) < (ll_delta + lp_delta + ll_delta_det):
                self.state.z[i] = z_prop
                self._basis_beta[i] = b_new_beta
                self._basis_gamma[i] = b_new_gamma
                self._edge_contrib[i] += delta_edge
                self._baseline_contrib[i] += delta_baseline
                self._acc["z_acc"] += 1

    def _update_z(self, it):
        self._update_z_single_row(it=it)



    def _update_rho(self) -> None:
        """Update rho parameter"""
        a = self.h.a_rho + self.state.r.sum()
        b = self.h.b_rho + np.prod(self.state.r.shape) - self.state.r.sum()
        self.state.rho = np.random.beta(a, b)


    def _update_tau_gamma(self):
        """Update tau_gamma with precomputed penalty matrix"""
        RtR = self.RtR_gamma + 1e-8 * np.eye(self.K_gamma)
        p_par = self.h.a_s - self.K_gamma / 2
        a = 2.0 * self.h.b_s
        
        for j in range(self.p):
            s = self.state.gamma[j]
            chi = s @ RtR @ s
            self.state.tau_gamma[j] = np.clip(gig_rvs(a, chi, p_par,rng=self.seed), 1e-2, 1e4)

        
    def _update_C(self) -> None:
        """Update C using multivariate normal draw based on variable residuals, excluding static variables."""
        tau_C = self.h.tau_C
        K_c = self.K_c
        d = len(self.var_indices)  # Number of variables used for C
        non_static_indices = np.where(~self._no_incoming)[0]  # Indices of non-static variables
        p_non_static = len(non_static_indices)  # Number of non-static variables
        residuals = np.zeros((self.T - 1, self.n, p_non_static))  # Residuals for non-static vars
        grad_pred = np.zeros((self.T - 1, self.n, p_non_static, d * K_c))  # Gradients w.r.t. C
        log_prior_z = np.zeros((self.T - 1, self.n))  # Store log prior for z_new

        a_beta, m_beta = 2.0, 2.0  # Mild U-shape to favor interior of [0,1]

        for t in range(self.T - 1):
            for i in range(self.n):
                if not (self.valid_time_mask[t, i] and self.valid_time_mask[t + 1, i]):
                    continue
                
                c_tensor = self._basis_C[i]  # Shape: (d * K_c,)
                c_tensor = additive_basis_row(self.V_norm[t, i, :], K=self.K_c, degree=3, t=self.t_time)
                zeta_mean = c_tensor @ self.state.C 
                zeta = np.random.normal(loc=zeta_mean, scale=self.state.eta[i])
                z_new = 1 / (1 + np.exp(-zeta)).item()  # Sigmoid transform to [0,1]
                # Compute log prior for z_new (Beta prior)
                m = (a_beta - 1) * np.log(z_new + 1e-10) + \
                                (m_beta - 1) * np.log1p(-z_new + 1e-10)
                log_prior_z[t, i] = m
                
                # Jacobian adjustment for MH (log|dz/d zeta| = log(z*(1-z)))
                jac_z = np.log(z_new * (1 - z_new) + 1e-10)
                
                # Compute basis functions for z_new
                b_beta = bspline_basis_single_row(z_new, K=self.K_beta, degree=self.degree, t=self.t_beta)
                b_gamma = bspline_basis_single_row(z_new, K=self.K_gamma, degree=self.degree, t=self.t_gamma)
                db_beta_dz = bspline_basis_derivative(z_new, K=self.K_beta, degree=self.degree, t=self.t_beta)
                db_gamma_dz = bspline_basis_derivative(z_new, K=self.K_gamma, degree=self.degree, t=self.t_gamma)

                X_pred = self.X_time[t + 1, i, non_static_indices]  # Observed non-static vars at t+1
                mu_pred = np.zeros(p_non_static)  # Predicted values for non-static vars
                dz_dzeta = z_new * (1 - z_new)

                for j_idx, j in enumerate(non_static_indices):
                    edge_sum = 0.0
                    for l in range(self.p):  # Use all vars (static or not) as predictors from t
                        if l != j and not self.forbidden[j, l]:
                            coeff = self.state.beta[j, l] * self.state.r[j, l]
                            if coeff.any():
                                # Use X_time[t, i, l] (previous time) for prediction
                                edge_sum += self.X_time[t, i, l] * (b_beta @ coeff)
                    mu_pred[j_idx] = (b_gamma @ self.state.gamma[j]) + edge_sum
                    residuals[t, i, j_idx] = X_pred[j_idx] - mu_pred[j_idx]

                    # Gradient of prediction w.r.t. C
                    grad_j = (db_gamma_dz @ self.state.gamma[j] * dz_dzeta) * c_tensor
                    for l in range(self.p):
                        if l != j and not self.forbidden[j, l]:
                            coeff = self.state.beta[j, l] * self.state.r[j, l]
                            if coeff.any():
                                grad_j += ((db_beta_dz @ coeff) * self.X_time[t, i, l] * dz_dzeta) * c_tensor
                    grad_pred[t, i, j_idx] = grad_j

        # Posterior covariance
        Sigma_inv = (1.0 / tau_C**2) * np.eye(d * K_c)  # Shape: (d * K_c, d * K_c)
        for t in range(self.T - 1):
            for i in range(self.n):
                if not (self.valid_time_mask[t, i] and self.valid_time_mask[t + 1, i]):
                    continue
                for j_idx, j in enumerate(non_static_indices):
                    Sigma_inv += (1.0 / self.state.sigma2[j]) * np.outer(grad_pred[t, i, j_idx], grad_pred[t, i, j_idx])

        try:
            L = np.linalg.cholesky(Sigma_inv + 1e-6 * np.eye(d * K_c))  # Add jitter
            Sigma_post = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(d * K_c)))
        except np.linalg.LinAlgError:
            print("LinAlgError in _update_C: Sigma_inv likely ill-conditioned")
            return  # Skip update to avoid crash

        # Posterior mean
        grad_ll = np.zeros(d * K_c)
        for t in range(self.T - 1):
            for i in range(self.n):
                if not (self.valid_time_mask[t, i] and self.valid_time_mask[t + 1, i]):
                    continue
                for j_idx, j in enumerate(non_static_indices):
                    grad_ll += (residuals[t, i, j_idx] / self.state.sigma2[j]) * grad_pred[t, i, j_idx]
                # Add prior contribution to gradient (Beta prior on z_new)
                grad_ll += (d_logp_dz * dz_dzeta) * c_tensor  # Remove / eta[i]

        mu_post = Sigma_post @ grad_ll
        self.state.C = st.multivariate_normal.rvs(mean=mu_post, cov=Sigma_post, random_state=self.rng)


    def _update_eta(self):
        """Update eta as subject-specific variances using longitudinal residuals"""
        non_static_indices = np.where(~self._no_incoming)[0]
        num_non_static = len(non_static_indices)
        
        for i in range(self.n):
            total_ss = 0.0
            num_obs = 0
            
            # Precompute basis evaluations for this subject (constant across time)
            basis_gamma_i = self._basis_gamma[i]  # (K_gamma,)
            basis_beta_i = self._basis_beta[i]    # (K_beta,)
            
            baseline_i = basis_gamma_i @ self.state.gamma.T  # (p,)
            
            for t in range(self.X_time.shape[0]):
                if not self.valid_time_mask[t, i]:
                    continue
                
                # Compute edge contributions for this time point and subject
                edge_i = np.zeros(self.p)
                for j in non_static_indices:
                    col = 0.0
                    for l in range(self.p):
                        if l == j or self.forbidden[j, l]:
                            continue
                        coeff = self.state.beta[j, l] * self.state.r[j, l]  # (K_beta,)
                        if np.any(coeff):
                            col += self.X_time[t, i, l] * (basis_beta_i @ coeff)
                    edge_i[j] = col
                
                # Residuals for non-static variables only (static have no incoming, residuals ignored)
                resid_i = self.X_time[t, i, :] - baseline_i - edge_i
                total_ss += np.sum(resid_i[non_static_indices] ** 2)
                num_obs += num_non_static
            
            if num_obs == 0:
                # Fallback if no valid data (unlikely)
                self.state.eta[i] = st.invgamma.rvs(a=self.h.a_sigma, scale=self.h.b_sigma)
                continue
            
            shape = self.h.a_sigma + num_obs / 2
            scale = self.h.b_sigma + 0.5 * total_ss
            self.state.eta[i] = st.invgamma.rvs(a=shape, scale=scale)

    def _update_mu(self):
        """Update mu parameter using the inverse gamma distribution and residuals"""
        resid = self.X - self._baseline_contrib - self._edge_contrib
        shape = self.h.a_sigma + (self.n * self.p) / 2
        scale = self.h.b_sigma + 0.5 * np.sum(resid ** 2)
        self.state.mu = st.invgamma.rvs(a=shape, scale=scale)
        
        pass



    def _adapt_step_size(self):
        """Adaptive step size for MH"""
        if self._acc["z_prop"]:
            rate = self._acc["z_acc"] / self._acc["z_prop"]
            if rate > self.target_accept_rate:
                self.z_step *= 1 + self.step_adapt_rate
            else:
                self.z_step *= 1 - self.step_adapt_rate
            self.z_step = np.clip(self.z_step, 0.001, 0.5)


    def _print_acceptance(self, it: int):
        """Print acceptance statistics"""
        z_rate = self._acc["z_acc"] / max(1, self._acc["z_prop"])
        print(f"iter {it:>5} │ z‑accept = {z_rate:5.2%} │ z‑step = {self.z_step:.4f}")
        if it % 5 == 0:
            self._adapt_step_size()
        self._acc["z_prop"] = self._acc["z_acc"] = 0



    def run(self, n_iter: int = 2000, burnin: int = 500, thin: int = 5) -> List[StateBS]:
        """Run MCMC sampler with optimizations"""
        draws: List[StateBS] = []
        
        update_interval = 1
        
        for it in range(1, n_iter + 1):
            if it % update_interval == 0:
                self._BtB_beta = self._basis_beta.T @ self._basis_beta
                self._BtB_gamma = self._basis_gamma.T @ self._basis_gamma
            
            self._update_sigma2()
            self._update_gamma()
            self._update_tau_gamma()
            self._update_beta_and_r_blocked()
            self._update_rho()
            self._update_beta()
            self._update_tau2_mh()
            self._update_z(it=it)
            # self._update_eta()
            # self._update_C()
            
            if it >= burnin and (it - burnin) % thin == 0:
                

                draws.append(copy.deepcopy(self.state))
            
            if (it % self.print_every) == 0:
                self._print_acceptance(it)

            
            assert not self.state.r[self._no_incoming].any(), "incoming edge into a static node!"
        
        return draws
    