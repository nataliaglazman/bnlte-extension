from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import scipy.stats as st
from scipy.interpolate import BSpline
from scipy.stats import geninvgauss, truncnorm
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
from scipy.linalg import cho_solve




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
    z_time: np.ndarray            # (T, n)
    phi: np.ndarray               # (n,)
    sigma_epsilon: float          # global transition variance

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
            nu_epsilon: float = 0.001,
            lambda_epsilon: float = 0.001,
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
            self.nu_epsilon = nu_epsilon
            self.lambda_epsilon = lambda_epsilon

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
        var_indices: Optional[List[int]] = None
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
        self.T = self.X_time.shape[0]  # Dynamic T from data
        self.valid_time_mask = ~np.isnan(self.X_time).any(axis=2)  # Shape (T, n), True where data is valid
        print('Valid time mask shape:', self.valid_time_mask.shape)
        X = X[0]
        self.X = np.asarray(X, np.float64)  
        self.n, self.p = self.X.shape

        self.corr_matrix = np.corrcoef(self.X, rowvar=False)

        pca = PCA(n_components=1)
        self.pc1 = pca.fit_transform(self.X)[:, 0]
        self.pc_mean = np.mean(self.pc1)
        self.pc_direction = -np.sign(np.mean(pca.components_[0]))  # Optional: Adjust sign if you know the expected direction (e.g., positive if most biomarkers increase with progression)
        print(f"PCA direction: {self.pc_direction}")
        self.pc1 *= self.pc_direction  # Flip PC1 if needed to ensure "positive" direction aligns with expected progression

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

        self.K_c = 5
        self.degree = degree

        self.binary_indices = [1, 2]
        self.continuous_indices = [0, 3]
        self.K_c_cont = self.K_c  # Number of B-spline basis functions for continuous covariates


        sigma3 =  st.gamma.rvs(a=self.a_t, scale=self.b_t)
        self.var_indices = var_indices  # Indices of the 3 variables to use for time modeling
        z0 = np.random.uniform(0.1, 0.9, self.n)
        # c0 = np.random.rand(3, self.K_beta)
        # c0 = np.random.rand(len(self.var_indices) * self.K_c)
        c0 = np.random.randn(len(self.binary_indices) + len(self.continuous_indices) * self.K_c_cont) * 0.01  # Shape: (d_binary + d_cont * K_c_cont,)

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
        z_time0 = np.zeros((self.T, self.n))
        phi0 = np.random.uniform(0.01, 0.1, self.n)
        sigma_epsilon0 = st.invgamma.rvs(self.h.nu_epsilon, scale=self.h.lambda_epsilon)
        for i in range(self.n):
            z_time0[:, i] = np.cumsum(np.concatenate(([z0[i]], phi0[i] * np.ones(self.T - 1))))
        z_time0 = np.clip(z_time0, 0.1, 0.9)
        self.state = StateBS(sigma2_0, beta0, gamma0, r0, z0, tau0, sigma3, rho, c0, eta0, z_time0, phi0, sigma_epsilon0)
        
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
        self._basis_C = np.array([self.mixed_covariate_row(self.V_norm[0, i, :]) for i in range(self.n)])  # Shape: (n, d_binary + d_cont * K_c_cont)
        self.tau_C_binary = 1.0
        self.tau_C_cont = 1.0
        


        print("Basis C shape:", self._basis_C.shape)  # Should be (n, K_c^3)


        self.RtR_gamma = _second_diff_penalty(self.K_gamma)
        

        self._chol_cache = {}

    def mixed_covariate_row(self, v: np.ndarray) -> np.ndarray:
        """Compute covariate row: linear for binary, B-splines for continuous."""
        # Binary covariates: direct inclusion (linear term)
        binary_terms = v[self.binary_indices]  # Shape: (len(binary_indices),)
        
        # Continuous covariates: B-spline basis
        cont_terms = []
        for j in self.continuous_indices:
            v_j = v[j]
            basis_row = bspline_basis_single_row(v_j, K=self.K_c_cont, degree=self.degree, t=self.t_time)
            cont_terms.append(basis_row)
        cont_terms = np.concatenate(cont_terms) if cont_terms else np.array([])  # Shape: (len(volume_indices) * K_c_cont,)
        
        # Combine
        return np.concatenate([binary_terms, cont_terms])  # Shape: (d_binary + d_cont * K_c_cont,)


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
        dalign = lambda_align * ((z_new - z_old) * (self.pc1[i] - self.pc_mean))

        return dalign + dbeta + dcoul

    def _update_z_time(self, it) -> None:
        """Optimized z update for all time points, with transition prior for t >= 1"""
        sigma2 = self.state.sigma2
        inv_sigma2 = 1.0 / sigma2
        active_mask = ~self._no_incoming
        
        # Update for t=0 (baseline, same as original _update_z)
        for i in range(self.n):
            self._acc["z_prop"] += 1
            z_old = self.state.z_time[0, i]
            z_prop = (z_old + np.random.normal(scale=self.z_step)) % 2.0
            if z_prop > 1:
                z_prop = 2 - z_prop

            b_old_beta = self._basis_beta[i]
            b_new_beta = bspline_basis_single_row(z_prop, K=self.K_beta, t=self.t_beta)
            b_old_gamma = self._basis_gamma[i]
            b_new_gamma = bspline_basis_single_row(z_prop, K=self.K_gamma, t=self.t_gamma)

            delta_edge = np.zeros(self.p)
            
            for j in np.where(active_mask)[0]:
                edge_sum = 0.0
                for l in range(self.p):
                    if l != j and not self.forbidden[j, l]:
                        coeff = self.state.beta[j, l] * self.state.r[j, l]
                        if coeff.any():
                            edge_sum += self.X[i, l] * ((b_new_beta @ coeff) - (b_old_beta @ coeff))
                delta_edge[j] = edge_sum

            delta_baseline = (b_new_gamma - b_old_gamma) @ self.state.gamma.T
            resid_old = self.X[i] - self._baseline_contrib[i] - self._edge_contrib[i]
            resid_new = resid_old - delta_baseline - delta_edge

            lp_delta = self._log_prior_z_delta(i, z_old, z_prop)
            
            B_old = self._B_from_basis_row(b_old_beta)
            B_new = self._B_from_basis_row(b_new_beta)
            _, logdet_old = np.linalg.slogdet(np.eye(self.p) - B_old)
            _, logdet_new = np.linalg.slogdet(np.eye(self.p) - B_new)
            
            ll_delta_det = logdet_new - logdet_old
            ll_delta = 0.5 * np.sum((resid_old[active_mask] ** 2 - resid_new[active_mask] ** 2) * inv_sigma2[active_mask])

            if np.log(np.random.rand()) < (ll_delta + lp_delta + ll_delta_det):
                self.state.z_time[0, i] = z_prop
                self.state.z[i] = z_prop  # Sync baseline z
                self._basis_beta[i] = b_new_beta
                self._basis_gamma[i] = b_new_gamma
                self._edge_contrib[i] += delta_edge
                self._baseline_contrib[i] += delta_baseline
                self._acc["z_acc"] += 1

        # Update for t >= 1
        for t in range(1, self.T):
            for i in range(self.n):
                if not self.valid_time_mask[t, i]:
                    continue
                self._acc["z_prop"] += 1
                z_current = self.state.z_time[t, i]
                z_prop = (z_current + np.random.normal(scale=self.z_step)) % 2.0
                if z_prop > 1:
                    z_prop = 2 - z_prop
                z_prop = np.clip(z_prop, 0.01, 0.99)

                b_old_beta = bspline_basis_single_row(z_current, K=self.K_beta, t=self.t_beta)
                b_new_beta = bspline_basis_single_row(z_prop, K=self.K_beta, t=self.t_beta)
                b_old_gamma = bspline_basis_single_row(z_current, K=self.K_gamma, t=self.t_gamma)
                b_new_gamma = bspline_basis_single_row(z_prop, K=self.K_gamma, t=self.t_gamma)

                delta_baseline = b_new_gamma @ self.state.gamma.T - b_old_gamma @ self.state.gamma.T
                delta_edge = np.zeros(self.p)
                for j in np.where(active_mask)[0]:
                    edge_sum = 0.0
                    for l in range(self.p):
                        if l != j and not self.forbidden[j, l]:
                            coeff = self.state.beta[j, l] * self.state.r[j, l]
                            if coeff.any():
                                edge_sum += self.X_time[t, i, l] * ((b_new_beta - b_old_beta) @ coeff)
                    delta_edge[j] = edge_sum

                resid_old = np.zeros(self.p)
                for j in np.where(active_mask)[0]:
                    baseline_old = b_old_gamma @ self.state.gamma[j]
                    edge_old = 0.0
                    for l in range(self.p):
                        if l != j and not self.forbidden[j, l]:
                            coeff = self.state.beta[j, l] * self.state.r[j, l]
                            if coeff.any():
                                edge_old += self.X_time[t, i, l] * (b_old_beta @ coeff)
                    resid_old[j] = self.X_time[t, i, j] - baseline_old - edge_old

                resid_new = resid_old - delta_baseline - delta_edge

                ll_delta = 0.5 * np.sum((resid_old[active_mask] ** 2 - resid_new[active_mask] ** 2) * inv_sigma2[active_mask])

                # Transition prior delta
                z_prev = self.state.z_time[t - 1, i]
                phi_i = self.state.phi[i]
                sigma_e = np.sqrt(self.state.sigma_epsilon)
                lp_old = st.norm.logpdf(z_current, loc=z_prev + phi_i, scale=sigma_e)
                lp_new = st.norm.logpdf(z_prop, loc=z_prev + phi_i, scale=sigma_e)
                lp_delta = lp_new - lp_old

                # Logdet delta
                B_old = self._B_from_basis_row(b_old_beta)
                B_new = self._B_from_basis_row(b_new_beta)
                _, logdet_old = np.linalg.slogdet(np.eye(self.p) - B_old)
                _, logdet_new = np.linalg.slogdet(np.eye(self.p) - B_new)
                ll_delta_det = logdet_new - logdet_old

                if np.log(self.rng.random()) < (ll_delta + lp_delta + ll_delta_det):
                    self.state.z_time[t, i] = z_prop
                    self._acc["z_acc"] += 1



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

        

    def _update_C(self):
        """Update C with mixed parameterization and type-specific regularization."""
        d_binary = len(self.binary_indices)
        d_cont = len(self.continuous_indices)
        d_total = d_binary + d_cont * self.K_c_cont
        Sigma_inv = np.zeros((d_total, d_total))
        
        # Regularization: ridge prior with type-specific penalties
        for j in range(d_total):
            if j < d_binary:
                Sigma_inv[j, j] = 1.0 / self.tau_C_binary  # Weaker penalty for binary
            else:
                Sigma_inv[j, j] = 1.0 / self.tau_C_cont  # Stronger penalty for volumes
        
        grad_ll = np.zeros(d_total)
        for i in range(self.n):
            c_tensor = self.mixed_covariate_row(self.V_norm[0, i, :])  # Use t=0 or latest valid time
            outer_ct = np.outer(c_tensor, c_tensor) / self.state.eta[i]
            Sigma_inv += outer_ct
            grad_ll += (self.state.phi[i] / self.state.eta[i]) * c_tensor
        
        try:
            L = np.linalg.cholesky(Sigma_inv + 1e-6 * np.eye(d_total))
            mu_post = cho_solve((L, True), grad_ll)
            Sigma_post = cho_solve((L, True), np.eye(d_total))
            self.state.C = st.multivariate_normal.rvs(mean=mu_post, cov=Sigma_post, random_state=self.rng)
        except np.linalg.LinAlgError:
            print("LinAlgError in _update_C: Sigma_inv ill-conditioned")
            return
        
    def _update_eta(self):
        """Update eta as subject-specific variances for phi using prior residuals.
        
        Uses averaged covariates across valid time points for stability.
        Stronger prior and scaled residual to prevent large eta_i.
        """
        for i in range(self.n):
            valid_times = np.where(self.valid_time_mask[:, i])[0]
            if len(valid_times) == 0:
                self.state.eta[i] = st.invgamma.rvs(a=self.h.nu_eta, scale=self.h.lambda_eta, random_state=self.rng)
                continue
            
            # Average c_tensor over valid time points
            c_tensor = np.mean([self.mixed_covariate_row(self.V_norm[t, i, :]) for t in valid_times], axis=0)
            mu_prior = c_tensor @ self.state.C
            residual = (self.state.phi[i] - mu_prior) ** 2
            
            # Stronger prior and dampened residual
            shape = self.h.nu_eta + 1.0  # Increased for stability (e.g., nu_eta=3)
            scale = self.h.lambda_eta + 0.1 * residual  # Scale down residual impact
            self.state.eta[i] = st.invgamma.rvs(a=shape, scale=scale, random_state=self.rng)

    # def _update_eta(self):
    #     """Update eta as subject-specific variances for phi using prior residuals.
        
    #     For each subject i, eta_i is the variance in phi_i ~ N(c_tensor @ C, eta_i).
    #     Uses the most recent valid time point's covariates for c_tensor, matching _update_phi.
    #     Samples eta_i from an inverse-gamma distribution based on the residual (phi_i - mu_prior)^2.
    #     """
    #     for i in range(self.n):
    #         # Find the most recent valid time point for subject i
    #         valid_times = np.where(self.valid_time_mask[:, i])[0]
    #         if len(valid_times) == 0:
    #             # Fallback: sample from prior if no valid data
    #             self.state.eta[i] = 1e-8
    #             continue
            
    #         # Use the most recent valid time point's covariates
    #         t_latest = valid_times[-1]
    #         # c_tensor = additive_basis_row(
    #         #     self.V_norm[t_latest, i, :], K=self.K_c, degree=self.degree, t=self.t_time
    #         # )
    #         c_tensor = self.mixed_covariate_row(self.V_norm[t_latest, i, :])  # Use t=0 or latest valid time
            
    #         # Compute prior mean for phi_i
    #         mu_prior = c_tensor @ self.state.C
            
    #         # Compute residual: (phi_i - mu_prior)^2
    #         residual = (self.state.phi[i] - mu_prior) ** 2
            
    #         # Update eta_i using inverse-gamma conjugate prior
    #         shape = self.h.nu_eta + 0.5  # One observation (phi_i)
    #         scale = self.h.lambda_eta + 0.5 * residual
    #         # print('Updating eta for subject', i, 'with shape', shape, 'and scale', scale, 'residual', residual, 'mu_prior', mu_prior, 'phi', self.state.phi[i])
    #         self.state.eta[i] = st.invgamma.rvs(a=shape, scale=scale, random_state=self.rng)


    # def _update_eta(self):
    #         """Update eta as subject-specific variances using longitudinal residuals with per-time z"""
    #         active_mask = ~self._no_incoming
    #         num_active = np.sum(active_mask)
            
    #         for i in range(self.n):
    #             total_ss = 0.0
    #             num_obs = 0
                
    #             for t in range(self.T):
    #                 if not self.valid_time_mask[t, i]:
    #                     continue
                    
    #                 z_ti = self.state.z_time[t, i]
    #                 b_gamma = bspline_basis_single_row(z_ti, K=self.K_gamma, degree=self.degree, t=self.t_gamma)
    #                 b_beta = bspline_basis_single_row(z_ti, K=self.K_beta, degree=self.degree, t=self.t_beta)
                    
    #                 baseline_i = b_gamma @ self.state.gamma.T  # (p,)
                    
    #                 edge_i = np.zeros(self.p)
    #                 for j in np.where(active_mask)[0]:
    #                     col = 0.0
    #                     for l in range(self.p):
    #                         if l == j or self.forbidden[j, l]:
    #                             continue
    #                         coeff = self.state.beta[j, l] * self.state.r[j, l]  # (K_beta,)
    #                         if np.any(coeff):
    #                             col += self.X_time[t, i, l] * (b_beta @ coeff)
    #                     edge_i[j] = col
                    
    #                 resid_i = self.X_time[t, i, :] - baseline_i - edge_i
    #                 total_ss += np.sum(resid_i[active_mask] ** 2)
    #                 num_obs += num_active
                
    #             if num_obs == 0:
    #                 # Fallback if no valid data
    #                 self.state.eta[i] = st.invgamma.rvs(a=self.h.nu_eta, scale=self.h.lambda_eta)
    #                 continue
                
    #             shape = self.h.nu_eta + num_obs / 2
    #             scale = self.h.lambda_eta + 0.5 * total_ss
            
    #             self.state.eta[i] = st.invgamma.rvs(a=shape, scale=scale)

    # def _update_phi(self):
    #     """Update phi using mixed covariate parameterization."""
    #     for i in range(self.n):
    #         deltas = []
    #         for t in range(1, self.T):
    #             if self.valid_time_mask[t - 1, i] and self.valid_time_mask[t, i]:
    #                 deltas.append(self.state.z_time[t, i] - self.state.z_time[t - 1, i])
            
    #         valid_times = np.where(self.valid_time_mask[:, i])[0]
    #         if len(valid_times) == 0:
    #             c_tensor = self.mixed_covariate_row(self.V_norm[0, i, :])
    #             mu_prior = c_tensor @ self.state.C
    #             sigma_prior = np.sqrt(self.state.eta[i])
    #             self.state.phi[i] = truncnorm.rvs(a=0, b=np.inf, loc=mu_prior, scale=sigma_prior, random_state=self.rng)
    #             continue
            
    #         t_latest = valid_times[-1]
    #         c_tensor = self.mixed_covariate_row(self.V_norm[t_latest, i, :])
    #         mu_prior = c_tensor @ self.state.C
    #         sigma_prior = np.sqrt(self.state.eta[i])
            
    #         if not deltas:
    #             self.state.phi[i] = truncnorm.rvs(a=0, b=np.inf, loc=mu_prior, scale=sigma_prior, random_state=self.rng)
    #             continue
            
    #         mean_delta = np.mean(deltas)
    #         num_d = len(deltas)
    #         prec_like = num_d / self.state.sigma_epsilon
    #         prec_prior = 1 / sigma_prior**2
    #         mu_post = (prec_prior * mu_prior + prec_like * mean_delta) / (prec_prior + prec_like)
    #         sigma_post = np.sqrt(1 / (prec_prior + prec_like))
    #         self.state.phi[i] = truncnorm.rvs(a=0, b=np.inf, loc=mu_post, scale=sigma_post, random_state=self.rng)

    def _update_phi(self):
        """Update phi using multiplicative mixed covariate parameterization with diagnostics."""
        for i in range(self.n):
            deltas = []
            for t in range(1, self.T):
                if self.valid_time_mask[t - 1, i] and self.valid_time_mask[t, i]:
                    delta = self.state.z_time[t, i] - self.state.z_time[t - 1, i]
                    deltas.append(delta)
            
            valid_times = np.where(self.valid_time_mask[:, i])[0]
            if len(valid_times) == 0:
                c_tensor = self.mixed_covariate_row(self.V_norm[0, i, :])
                mu_prior = c_tensor @ self.state.C
                sigma_prior = np.sqrt(self.state.eta[i])
                phi_new = 1e-8
                # print(f"Subject {i}: No valid times, mu_prior={mu_prior:.4f}, sigma_prior={sigma_prior:.4f}, phi={phi_new:.4f}")
                self.state.phi[i] = phi_new
                continue
            
            t_latest = valid_times[-1]
            c_tensor = self.mixed_covariate_row(self.V_norm[t_latest, i, :])
            mu_prior = c_tensor @ self.state.C
            sigma_prior = np.sqrt(self.state.eta[i])
            
            if not deltas:
                # phi_new = truncnorm.rvs(a=0, b=np.inf, loc=mu_prior, scale=sigma_prior, random_state=self.rng)
                # print(f"Subject {i}: No deltas, mu_prior={mu_prior:.4f}, sigma_prior={sigma_prior:.4f}, phi={phi_new:.4f}")
                self.state.phi[i] = 1e-8
                continue
            
            mean_delta = np.mean(deltas)
            num_d = len(deltas)
            prec_like = num_d / self.state.sigma_epsilon
            prec_prior = 1 / sigma_prior**2
            mu_post = (prec_prior * mu_prior + prec_like * mean_delta) / (prec_prior + prec_like)
            sigma_post = np.sqrt(1 / (prec_prior + prec_like))
            phi_new = truncnorm.rvs(a=0, b=np.inf, loc=mu_post, scale=sigma_post, random_state=self.rng)
            # print(f"Subject {i}: mean_delta={mean_delta:.4f}, num_d={num_d}, mu_prior={mu_prior:.4f}, sigma_prior={sigma_prior:.4f}, mu_post={mu_post:.4f}, sigma_post={sigma_post:.4f}, phi={phi_new:.4f}")
            self.state.phi[i] = phi_new

    def _update_sigma_epsilon(self) -> None:
        """Update sigma_epsilon using transition residuals."""
        total_ss = 0.0
        num_trans = 0
        for t in range(1, self.T):
            for i in range(self.n):
                if self.valid_time_mask[t - 1, i] and self.valid_time_mask[t, i]:
                    delta = self.state.z_time[t, i] - self.state.z_time[t - 1, i] - self.state.phi[i]
                    total_ss += delta ** 2
                    num_trans += 1
        
        shape = self.h.nu_epsilon + num_trans / 2
        scale = self.h.lambda_epsilon + 0.5 * total_ss
        self.state.sigma_epsilon = st.invgamma.rvs(a=shape, scale=scale)

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
            self._update_z_time(it)
            self._update_phi()
            self._update_sigma_epsilon()
            self._update_eta()
            self._update_C()
            
            if it >= burnin and (it - burnin) % thin == 0:
                

                draws.append(copy.deepcopy(self.state))
            
            if (it % self.print_every) == 0:
                self._print_acceptance(it)

            
            assert not self.state.r[self._no_incoming].any(), "incoming edge into a static node!"
        
        return draws