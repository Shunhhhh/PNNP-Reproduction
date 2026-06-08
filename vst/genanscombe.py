"""
Generalized Anscombe variance-stabilizing transformation and its inverses.

Implements the Poisson-Gaussian noise model:

    z = alpha * p + n

where p ~ Poisson(y)  and  n ~ N(g, sigma^2).

References
----------
[1] J.L. Starck, F. Murtagh, and A. Bijaoui, Image Processing and Data
    Analysis, Cambridge University Press, Cambridge, 1998.
[2] M. Makitalo and A. Foi, "Optimal inversion of the generalized Anscombe
    transformation for Poisson-Gaussian noise", IEEE Trans. Image Process.,
    doi:10.1109/TIP.2012.2202675.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d

from .anscombe import forward as anscombe_forward
from .anscombe import inverse_exact_unbiased as anscombe_inverse_exact_unbiased
from .data.data_loader import load_genanscombe_vectors


def forward(z, sigma, alpha=None, g=None):
    """
    Generalized Anscombe variance-stabilizing transformation.

    Parameters
    ----------
    z : ndarray
        Noisy observation following the Poisson-Gaussian model.
    sigma : float
        Standard deviation of the Gaussian noise component.
    alpha : float, optional
        Positive scaling factor of the Poisson component (default 1).
    g : float, optional
        Mean of the Gaussian noise component (default 0).

    Returns
    -------
    fz : ndarray
        Variance-stabilized data (variance approximately 1).

    Notes
    -----
    f(z) = 2/alpha * sqrt(max(0, alpha*z + 3/8*alpha^2 + sigma^2 - alpha*g))
    """
    if g is None:
        g = 0.0
    if alpha is None:
        alpha = 1.0

    fz = (2.0 / alpha
          * np.sqrt(np.maximum(
              0, alpha * z + (3.0 / 8.0) * alpha ** 2 + sigma ** 2 - alpha * g
          )))
    return fz


def inverse_closed_form(D, sigma, alpha=None, g=None):
    """
    Closed-form approximation of the exact unbiased inverse of the
    Generalized Anscombe transformation.

    Parameters
    ----------
    D : ndarray
        Filtered (denoised) signal after variance stabilization.
    sigma : float
        Standard deviation of the Gaussian noise component.
    alpha : float, optional
        Positive scaling factor of the Poisson component (default 1).
    g : float, optional
        Mean of the Gaussian noise component (default 0).

    Returns
    -------
    exact_inverse : ndarray
        Approximate unbiased estimate of the original intensity.

    Notes
    -----
    Uses the series expansion from [1] (same as the standard Anscombe
    closed-form inverse) combined with parameter rescaling.

    References
    ----------
    [1] M. Makitalo and A. Foi, "A closed-form approximation of the exact
    unbiased inverse of the Anscombe variance-stabilizing transformation",
    IEEE Trans. Image Process., vol. 20, no. 9, pp. 2697-2698, Sep. 2011.
    """
    alpha_provided = alpha is not None
    g_provided = g is not None

    # Normalise sigma if alpha is provided (matches MATLAB behaviour)
    if alpha_provided:
        sigma = sigma / alpha

    sqrt_3_over_2 = np.sqrt(3.0 / 2.0)
    exact_inverse = ((D / 2.0) ** 2
                     + 1.0 / 4.0 * sqrt_3_over_2 * D ** (-1)
                     - 11.0 / 8.0 * D ** (-2)
                     + 5.0 / 8.0 * sqrt_3_over_2 * D ** (-3)
                     - 1.0 / 8.0
                     - sigma ** 2)
    exact_inverse = np.maximum(0, exact_inverse)

    # Reverse the initial variable change
    if alpha_provided:
        exact_inverse = exact_inverse * alpha
    if g_provided:
        exact_inverse = exact_inverse + g

    return exact_inverse


def inverse_exact_unbiased(D, sigma, alpha=None, g=None, data_dir=None):
    """
    Exact unbiased inverse of the Generalized Anscombe transformation
    via pre-computed look-up tables.

    Parameters
    ----------
    D : ndarray
        Filtered (denoised) signal after variance stabilization.
    sigma : float
        Standard deviation of the Gaussian noise component.
    alpha : float, optional
        Positive scaling factor of the Poisson component (default 1).
    g : float, optional
        Mean of the Gaussian noise component (default 0).
    data_dir : str, optional
        Path to the directory containing GenAnscombe_vectors.mat.

    Returns
    -------
    exact_inverse : ndarray
        Exact unbiased estimate of the original intensity.

    Notes
    -----
    For sigma > max(sigmas): uses Anscombe exact unbiased inverse minus
    sigma^2 (asymptotic correction).
    For sigma > 0:   2-D interpolation in (sigma, Ez) domain, then 1-D
                      interpolation using the resulting Efz curve.
    For sigma == 0:  standard Anscombe exact unbiased inverse.
    For sigma < 0:   raises ValueError.

    Requires GenAnscombe_vectors.mat and Anscombe_vectors.mat.
    """
    alpha_provided = alpha is not None
    g_provided = g is not None

    # Normalise sigma if alpha is provided
    if alpha_provided:
        sigma = sigma / alpha

    # Load the pre-computed tables
    Efzmatrix, Ez, sigmas = load_genanscombe_vectors(data_dir)

    if sigma < 0:
        raise ValueError('sigma must be non-negative!')

    if sigma > np.max(sigmas):
        # Very large sigma: use Anscombe exact unbiased with -sigma^2 offset
        exact_inverse = anscombe_inverse_exact_unbiased(D, data_dir) - sigma ** 2
        exact_inverse = np.maximum(0, exact_inverse)

    elif sigma > 0:
        # Set up a 2-D interpolator for the (sigma, Ez) -> Efz mapping.
        # Efzmatrix has shape (len(Ez), len(sigmas)) in MATLAB.
        # RegularGridInterpolator expects values.shape = (len(x), len(y), ...)
        # where x = sigmas (column dim), y = Ez (row dim).
        interp_2d = RegularGridInterpolator(
            (sigmas, Ez), Efzmatrix.T,   # .T -> (len(sigmas), len(Ez))
            bounds_error=False, fill_value=None
        )

        # Query Efz at the given sigma for all Ez values
        query = np.column_stack([np.full_like(Ez, sigma), Ez])
        Efz = interp_2d(query)  # shape (len(Ez),)

        # 1-D lookup: D -> exact_inverse using (Efz, Ez)
        interp_1d = interp1d(Efz, Ez, kind='linear',
                             bounds_error=False,
                             fill_value='extrapolate')
        exact_inverse = interp_1d(D)

        # Outside the pre-computed domain, fall back to asymptotic
        large_mask = D > np.max(Efz)
        asymptotic = anscombe_inverse_exact_unbiased(D, data_dir) - sigma ** 2
        exact_inverse[large_mask] = asymptotic[large_mask]

        small_mask = D < np.min(Efz)
        exact_inverse[small_mask] = 0.0

    else:  # sigma == 0
        exact_inverse = anscombe_inverse_exact_unbiased(D, data_dir)

    # Reverse the initial variable change
    if alpha_provided:
        exact_inverse = exact_inverse * alpha
    if g_provided:
        exact_inverse = exact_inverse + g

    return exact_inverse