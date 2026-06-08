"""
Standard Anscombe variance-stabilizing transformation and its inverses.

Reference
---------
Anscombe, F.J., "The transformation of Poisson, binomial and negative-binomial
data", Biometrika, vol. 35, no. 3/4, pp. 246-254, Dec. 1948.
"""

import numpy as np
from scipy.interpolate import interp1d

from .data.data_loader import load_anscombe_vectors, load_mmse_curves


def forward(z):
    """
    Apply the Anscombe variance-stabilizing transformation.

    Parameters
    ----------
    z : ndarray
        Noisy Poisson-distributed data.

    Returns
    -------
    transformed : ndarray
        Variance-stabilized data (variance approximately 1).

    Notes
    -----
    f(z) = 2 * sqrt(z + 3/8)
    """
    return 2.0 * np.sqrt(z + 3.0 / 8.0)


def inverse_asympt_unbiased(D):
    """
    Asymptotically unbiased inverse of the Anscombe transformation.

    Parameters
    ----------
    D : ndarray
        Filtered (denoised) signal after variance stabilization.

    Returns
    -------
    asymptotic : ndarray
        Asymptotically unbiased estimate of the original intensity.

    Notes
    -----
    For D < 2*sqrt(3/8) the result is set to 0.
    """
    asymptotic = (D / 2.0) ** 2 - 1.0 / 8.0
    asymptotic[D < 2.0 * np.sqrt(3.0 / 8.0)] = 0.0
    return asymptotic


def inverse_closed_form(D):
    """
    Closed-form approximation of the exact unbiased inverse.

    Parameters
    ----------
    D : ndarray
        Filtered (denoised) signal after variance stabilization.

    Returns
    -------
    exact_inverse : ndarray
        Approximate unbiased estimate via a rational series expansion.

    Notes
    -----
    Uses the series expansion from [1] without needing look-up tables.

    References
    ----------
    [1] M. Makitalo and A. Foi, "A closed-form approximation of the exact
    unbiased inverse of the Anscombe variance-stabilizing transformation",
    IEEE Trans. Image Process., vol. 20, no. 9, pp. 2697-2698, Sep. 2011.
    """
    sqrt_3_over_2 = np.sqrt(3.0 / 2.0)
    exact_inverse = ((D / 2.0) ** 2
                     + 1.0 / 4.0 * sqrt_3_over_2 * D ** (-1)
                     - 11.0 / 8.0 * D ** (-2)
                     + 5.0 / 8.0 * sqrt_3_over_2 * D ** (-3)
                     - 1.0 / 8.0)
    exact_inverse = np.maximum(0, exact_inverse)
    return exact_inverse


def inverse_exact_unbiased(D, data_dir=None):
    """
    Exact unbiased inverse of the Anscombe transformation via look-up table.

    Parameters
    ----------
    D : ndarray
        Filtered (denoised) signal after variance stabilization.
    data_dir : str, optional
        Path to the directory containing Anscombe_vectors.mat.

    Returns
    -------
    exact_inverse : ndarray
        Exact unbiased estimate.

    Notes
    -----
    Uses pre-computed expectations Efz = E[f(z) | y] and Ez = E[z | y] = y.
    For large D (outside the pre-computed domain) the asymptotically unbiased
    inverse is used instead.  For very small D the result is set to 0.

    References
    ----------
    [1] M. Makitalo and A. Foi, "Optimal inversion of the Anscombe
    transformation in low-count Poisson image denoising",
    IEEE Trans. Image Process., vol. 20, no. 1, pp. 99-109, Jan. 2011.
    """
    Efz, Ez = load_anscombe_vectors(data_dir)

    # Asymptotically unbiased inverse (fallback for large values)
    asymptotic = (D / 2.0) ** 2 - 1.0 / 8.0

    # 1-D linear interpolation with extrapolation
    interp = interp1d(Efz, Ez, kind='linear', bounds_error=False,
                      fill_value='extrapolate')
    exact_inverse = interp(D)

    # For D larger than the pre-computed domain, revert to asymptotic
    large_mask = D > np.max(Efz)
    exact_inverse[large_mask] = asymptotic[large_mask]

    # For D smaller than 2*sqrt(3/8), set to 0
    small_mask = D < 2.0 * np.sqrt(3.0 / 8.0)
    exact_inverse[small_mask] = 0.0

    return exact_inverse


def inverse_mmse(D, stdD, data_dir=None):
    """
    MMSE (minimum mean-square error) inverse of the Anscombe transformation.

    Parameters
    ----------
    D : ndarray
        Filtered (denoised) signal after variance stabilization.
    stdD : ndarray or float
        Standard deviation of D (assumes D ~ N(E[f(z)|y], stdD^2)).
    data_dir : str, optional
        Path to the directory containing MMSEcurves.mat.

    Returns
    -------
    I_MMSE : ndarray
        MMSE estimate of the original intensity.

    Notes
    -----
    Uses a 2-D look-up table (sigmas x D1_values) and falls back to the
    exact unbiased inverse for large values, or log-domain extrapolation
    for very small values.

    References
    ----------
    [1] M. Makitalo and A. Foi, "Optimal inversion of the Anscombe
    transformation in low-count Poisson image denoising",
    IEEE Trans. Image Process., vol. 20, no. 1, pp. 99-109, Jan. 2011.
    """
    sigmas, D1_values, y_hats_values = load_mmse_curves(data_dir)
    n_sigmas = len(sigmas)
    n_D1 = len(D1_values)

    # Map stdD and D to indices in the table
    sigma_index = np.clip(
        np.interp(stdD, sigmas, np.arange(n_sigmas)),
        0, n_sigmas - 1
    )
    D_index = np.interp(D, D1_values, np.arange(n_D1))

    # ---- Large D: D_index > last index ----
    clip_large = D_index > n_D1 - 1
    D_index_clipped = D_index.copy()
    D_index_clipped[clip_large] = n_D1 - 1

    # Bi-linear interpolation into the 2-D table
    # MATLAB's interp2(Z, XI, YI) treats:
    #   Z as (nRows, nCols) where Y = 1:nRows (D1_values) and X = 1:nCols (sigmas)
    #   XI -> column query (sigma_index), YI -> row query (D_index)
    I_MMSE = _interp2_table(y_hats_values, D_index_clipped, sigma_index)

    # Correct large values with exact unbiased inverse
    if np.any(clip_large):
        I_MMSE[clip_large] += (inverse_exact_unbiased(D[clip_large], data_dir)
                               - inverse_exact_unbiased(
                                   np.asarray(D1_values[-1]), data_dir))

    # ---- Small D: D_index < 0 ----
    clip_small = D_index < 0
    D_index_clipped2 = D_index.copy()
    D_index_clipped2[clip_small] = 0

    if np.any(clip_small):
        # Logarithmic-domain linear extrapolation
        idx0 = 0
        idx1 = 1
        val0 = _interp2_table(y_hats_values,
                              np.full_like(sigma_index[clip_small], idx0),
                              sigma_index[clip_small])
        val1 = _interp2_table(y_hats_values,
                              np.full_like(sigma_index[clip_small], idx1),
                              sigma_index[clip_small])

        D_idx_clipped = D_index[clip_small]
        # Underflow: D_idx_clipped < 0 means we extrapolate below index 0
        # Weighted extrapolation in log domain
        log_val0 = np.log(np.maximum(val0, 1e-300))
        log_val1 = np.log(np.maximum(val1, 1e-300))
        extrap = np.exp(log_val0 * (1 - D_idx_clipped + 1)
                        + log_val1 * (D_idx_clipped - 1))
        I_MMSE[clip_small] = extrap

    return I_MMSE


def _interp2_table(Z, row_idx, col_idx):
    """
    Bi-linear interpolation into a 2-D table.

    This matches MATLAB's interp2 behaviour for regularly-gridded data.

    Parameters
    ----------
    Z : ndarray, shape (M, N)
        2-D look-up table.
    row_idx : ndarray
        Fractional row index (0-based).
    col_idx : ndarray
        Fractional column index (0-based).

    Returns
    -------
    values : ndarray
        Interpolated values.
    """
    row_floor = np.floor(row_idx).astype(int)
    col_floor = np.floor(col_idx).astype(int)
    row_ceil = np.minimum(row_floor + 1, Z.shape[0] - 1)
    col_ceil = np.minimum(col_floor + 1, Z.shape[1] - 1)

    wy = row_idx - row_floor
    wx = col_idx - col_floor

    v00 = Z[row_floor, col_floor]
    v10 = Z[row_ceil, col_floor]
    v01 = Z[row_floor, col_ceil]
    v11 = Z[row_ceil, col_ceil]

    return (v00 * (1 - wy) * (1 - wx)
            + v10 * wy * (1 - wx)
            + v01 * (1 - wy) * wx
            + v11 * wy * wx)
