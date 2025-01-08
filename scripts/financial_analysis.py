import numpy as np

def simulate_revenue(
    t, R, t_R, mu, sigma, R0, return_mean=False, seed=None, eps=1e-10
):
    """
    Conditional simulation of R(t) under GBM dynamics dR_t / R_t = mu dt + sigma dW_t,
    using log-transform and Gaussian process interpolation.

    Parameters
    ----------
    t : array_like
        Times at which we want R(t).
    R : array_like
        Known values of R at times t_R.
    t_R : array_like
        Times at which R is known.
    mu : float
        Drift parameter of GBM.
    sigma : float
        Volatility parameter of GBM.
    return_mean : bool, optional
        If True, return only the conditional mean of R(t).
        If False, return a single simulation (draw) from the conditional distribution.
    seed : int or None, optional
        Random seed for reproducible sampling. Default is None (no fixed seed).
    eps : float, optional
        Small jitter term for numerical stability in matrix inversion/cholesky.

    Returns
    -------
    R_t : ndarray
        Either the mean or a single simulated path of R(t) at the input times t,
        conditioned on the known R at t_R.
    """
    # Convert inputs to np arrays
    t = np.asarray(t, dtype=float)
    t_R = np.asarray(t_R, dtype=float)
    R = np.asarray(R, dtype=float)

    # Log of known R
    L_R = np.log(R)
    L0 = np.log(R0)

    # Mean function m(t) = (mu - 0.5*sigma^2)*t
    def mean_func(x):
        return L0 + (mu - 0.5 * sigma**2) * x

    # Covariance function: Cov(L(s), L(u)) = sigma^2 * min(s, u)
    def cov_func(x1, x2):
        return sigma**2 * np.minimum(x1[:, None], x2[None, :])

    # Precompute means
    m_R = mean_func(t_R)
    m_t = mean_func(t)

    # Precompute covariances
    K_RR = cov_func(t_R, t_R)
    K_tR = cov_func(t, t_R)
    K_Rt = K_tR.T
    K_tt = cov_func(t, t)

    # Add jitter to known-times covariance for numerical stability
    K_RR += eps * np.eye(len(t_R))

    # Invert K_RR
    K_RR_inv = np.linalg.inv(K_RR)

    # Conditional mean of L(t)
    L_t_mean = m_t + K_tR @ (K_RR_inv @ (L_R - m_R))

    if return_mean:
        # Return only the mean in R-space
        return np.exp(L_t_mean)
    else:
        # Conditional covariance of L(t)
        K_cond = K_tt - K_tR @ K_RR_inv @ K_Rt
        # Add jitter to K_cond for stability (optional, if needed)
        K_cond += eps * np.eye(len(t))

        # Sample from multivariate normal
        rng = np.random.default_rng(seed)
        z = rng.normal(size=len(t))
        L_t_sample = L_t_mean + np.linalg.cholesky(K_cond) @ z

        # Convert back to R-space
        return np.exp(L_t_sample)
