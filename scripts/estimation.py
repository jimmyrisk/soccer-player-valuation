import numpy as np
import pandas as pd
import scipy.stats.qmc as qmc
import numdifftools as nd
from tqdm import tqdm
from typing import Union
import contextlib
import sys
import os

# pymle imports
from pymle.core.Model import Model1D
from pymle.core.TransitionDensity import KesslerDensity
from pymle.fit.AnalyticalMLE import AnalyticalMLE
from pymle.fit.Estimator import EstimatedResult
from pymle.fit.Minimizer import ScipyMinimizer


# ----------------------------------------------------------------------
# Optional: suppress output inside a loop iteration (used for LHS trials)
# ----------------------------------------------------------------------
@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, 'w') as fnull:
        old_stdout = sys.stdout  # Save the current stdout
        sys.stdout = fnull       # Redirect stdout to devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout  # Restore original stdout


# ----------------------------------------------------------------------
# A custom Pearson model that matches the drift/diffusion used in code
# ----------------------------------------------------------------------
class Pearson2(Model1D):
    """
    Model for Pearson process
    Parameters: [kappa, mu, a, b, c]

    But here, we only really use [theta, pi_star, sigma].
    dX(t) = theta * (pi_star - X(t)) * dt + sqrt(2*theta*(...)) * dW_t
    """
    def __init__(self):
        super().__init__()

    def drift(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        # param[0] = theta, param[1] = pi_star
        return self._params[0] * (self._params[1] - x)

    def diffusion(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        # param[2] = sigma
        # Derivation specialized for "a = -sigma^2/(2*theta), b = sigma^2/(2*theta), c=0"
        sigma = self._params[2]
        theta = self._params[0]
        a = -sigma**2 / (2.0 * theta)
        b = sigma**2 / (2.0 * theta)
        c = 0.0
        return np.sqrt(2 * theta * (a * x**2 + b * x + c))

    def drift_t(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        return 0.        


# ----------------------------------------------------------------------
# Helper function to estimate parameters with a Hessian-based SE
# ----------------------------------------------------------------------
def _estimate_params2(estimator, params0: np.ndarray, likelihood) -> EstimatedResult:
    """
    Main estimation function that returns final params and standard errors
    """
    res = estimator._minimizer.minimize(
        function=likelihood,
        bounds=estimator._param_bounds,
        guess=params0
    )
    params = res.params
    final_like = -res.value

    # Hessian-based standard errors
    # (Add small diagonal to avoid singular Hessians)
    hessian = nd.Hessian(likelihood, method='central', order=2, step=1e-5)(params)
    H_inv = np.linalg.inv(hessian + np.diag([1e-8,1e-8,1e-8]))

    se = np.sqrt(H_inv.diagonal())


    result_obj = EstimatedResult(
        params=params,
        log_like=final_like,
        sample_size=len(estimator._sample) - 1
    )
    result_obj.se = se
    result_obj.hessian = hessian
    return result_obj


# ----------------------------------------------------------------------
# Main function for fitting the Pearson process to DataFrame columns 'pi' and 't'
# ----------------------------------------------------------------------
def pi_mle(pi_df: pd.DataFrame, n_samples=10, seed = None):
    """
    Fit a Pearson process to 'pi' with time steps from 't'.

    Args:
        pi_df (pd.DataFrame): Must contain columns 'pi' and 't'.
        n_samples (int): Number of LHS samples for searching initial guesses.  10 is default for speed, but should be increased for accuracy.

    Returns:
        (dict) A dictionary containing best-fit params, standard errors,
               and derived quantities.
    """
    # Sort to ensure time is in ascending order
    pi_df = pi_df.sort_values(by='t').reset_index(drop=True)

    # Observations
    sample = pi_df["pi"].values
    # Time steps (dt)
    h = pi_df['t'].diff()
    dt = h[1:].values  

    # Initialize the Pearson2 model
    model = Pearson2()

    # Bounds and "default" starting guess
    param_bounds = [(1.0, 30.0),    # theta
                    (0.01, 2/9),   # pi_star
                    (0.01, 0.5)]   # sigma
    default_guess = np.array([5.0, 1/9, 0.15])

    # We'll do an LHS over these bounds to find best MLE
    if seed is not None:
        generator = np.random.Generator(np.random.PCG64(seed))
        sampler = qmc.LatinHypercube(d=len(param_bounds), rng=generator)
    else:
        sampler = qmc.LatinHypercube(d=len(param_bounds))
    
    samples = sampler.random(n=n_samples)
    scaled_samples = qmc.scale(samples,
                               [b[0] for b in param_bounds],  # min
                               [b[1] for b in param_bounds])  # max

    # Container for results
    results = []

    # For each guess, we do MLE and store the best
    for param_guess in tqdm(scaled_samples, desc="LHS trials"):
        with suppress_output():
            # Create the estimator
            estimator = AnalyticalMLE(
                sample=sample,
                param_bounds=param_bounds,
                dt=dt,
                density=KesslerDensity(model)
            )
            # Adjust solver settings
            estimator._minimizer = ScipyMinimizer(method='L-BFGS-B', options={'maxiter': 1000, 'gtol': 1e-10, 'xtol': 1e-10, 'verbose': 1})
            est = _estimate_params2(estimator, param_guess, estimator.log_likelihood_negative)

        results.append({
            'log_likelihood': est.log_like,
            'theta':  est.params[0],
            'theta_se': est.se[0],
            'pi_star':   est.params[1],
            'pi_star_se': est.se[1],
            'sigma': est.params[2],
            'sigma_se': est.se[2],
        })

    # Convert to DataFrame and pick the best row
    df_results = pd.DataFrame(results)
    df_sorted = df_results.sort_values(by='log_likelihood', ascending=False)
    best = df_sorted.iloc[0]

    # Grab best-fit parameters
    theta_hat  = best['theta']
    pi0_hat    = best['pi_star']
    sigma_hat  = best['sigma']

    # The stationary variance for the Pearson process (based on this parameterization):
    #    stationary_var = pi0*(1 - pi0)*(sigma^2)/(2 * theta + sigma^2)
    stationary_var = pi0_hat * (1 - pi0_hat) * (sigma_hat**2) / (2*theta_hat + sigma_hat**2)
    sd_stationary  = np.sqrt(stationary_var)

    # A typical correlation measure used in OU-type processes over a 1-week horizon (1/52 of a year):
    #    cor = exp(-theta/52)
    cor_1week = np.exp(-theta_hat / 52.0)

    # Print results
    print("\n===========================")
    print("pi MLE Results:")
    print(f"theta     = {theta_hat:.6f}  ± {best['theta_se']:.6f}")
    print(f"pi_star      = {pi0_hat:.6f}    ± {best['pi_star_se']:.6f}")
    print(f"sigma     = {sigma_hat:.6f}  ± {best['sigma_se']:.6f}")
    print(f"stationary_var = {stationary_var:.6f}")
    print(f"sd (sqrt of var) = {sd_stationary:.6f}")
    print(f"cor(1wk)  = {cor_1week:.6f}")
    print("===========================\n")

    return {
        "theta": theta_hat,
        "pi_star": pi0_hat,
        "sigma": sigma_hat,
        "theta_se": best["theta_se"],
        "pi_star_se": best["pi_star_se"],
        "sigma_se": best["sigma_se"],
        "stationary_var": stationary_var,
        "sd_stationary": sd_stationary,
        "cor_1week": cor_1week,
        "log_likelihood": best["log_likelihood"],
    }
