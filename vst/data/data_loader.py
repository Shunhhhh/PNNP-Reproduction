"""
Load pre-computed lookup tables from .mat files.

These files are located in the parent project directory (invansc_v3/).
Users can also specify a custom path via the DATA_DIR environment variable
or by passing the path directly to the loader functions.
"""

import os
import numpy as np
from scipy.io import loadmat

# Default data directory: sibling to this package's parent
_DEFAULT_DATA_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..')
)


def _get_data_path(subpath, data_dir=None):
    """Resolve the full path to a .mat file."""
    base = data_dir or os.environ.get('INVANSC_DATA_DIR', _DEFAULT_DATA_DIR)
    return os.path.join(base, subpath)


def load_anscombe_vectors(data_dir=None):
    """
    Load the pre-computed Anscombe expectation vectors.

    Returns
    -------
    Efz : ndarray, shape (N,)
        E[f(z) | y] — expected value of the transformed signal.
    Ez : ndarray, shape (N,)
        E[z | y] = y — the original intensity values.
    """
    path = _get_data_path('Anscombe_vectors.mat', data_dir)
    data = loadmat(path, squeeze_me=True)
    return data['Efz'].astype(np.float64), data['Ez'].astype(np.float64)


def load_genanscombe_vectors(data_dir=None):
    """
    Load the pre-computed Generalized Anscombe expectation tables.

    Returns
    -------
    Efzmatrix : ndarray, shape (len(Ez), len(sigmas))
        2-D grid of E[f(z) | y] for each (Ez, sigma) pair.
    Ez : ndarray, shape (N,)
        E[z | y] = y — the original intensity values.
    sigmas : ndarray, shape (M,)
        Standard-deviation values used in the pre-computed grid.
    """
    path = _get_data_path('GenAnscombe_vectors.mat', data_dir)
    data = loadmat(path, squeeze_me=True)
    return (data['Efzmatrix'].astype(np.float64),
            data['Ez'].astype(np.float64),
            data['sigmas'].astype(np.float64))


def load_anscombe_lambda(data_dir=None):
    """
    Load expectation grid for the iterative VST Poisson denoising.

    Returns
    -------
    lambdaGridTimesE : ndarray, shape (len(yGrid), len(lambdaGrid))
        Pre-computed lambda * E[f(z) | y] for each (y, lambda).
    yGrid : ndarray, shape (N,)
        Grid of y values.
    lambdaGrid : ndarray, shape (M,)
        Grid of lambda values.
    """
    path = _get_data_path('Anscombe_lambda.mat', data_dir)
    data = loadmat(path, squeeze_me=True)
    return (data['lambdaGridTimesE'].astype(np.float64),
            data['yGrid'].astype(np.float64),
            data['lambdaGrid'].astype(np.float64))


def load_mmse_curves(data_dir=None):
    """
    Load the MMSE inverse lookup tables.

    Returns
    -------
    sigmas : ndarray, shape (M,)
        Standard-deviation dimension.
    D1_values : ndarray, shape (N,)
        Signal-value dimension.
    y_hats_values : ndarray, shape (len(sigmas), len(D1_values))
        2-D grid of MMSE estimates.
    """
    path = _get_data_path('MMSEcurves.mat', data_dir)
    data = loadmat(path, squeeze_me=True)
    return (data['sigmas'].astype(np.float64),
            data['D1_values'].astype(np.float64),
            data['y_hats_values'].astype(np.float64))


def load_params_from_qfun(data_dir=None):
    """
    Load the parameter-estimation function handle from paramsFromQfun.mat.

    The function handle stored in the .mat file is returned as a numpy
    object array.  In practice it must be evaluated or reconstructed
    manually; this loader returns the raw MATLAB object.

    Returns
    -------
    params_from_qfun : object
        The MATLAB function handle (callable in MATLAB only).
    """
    path = _get_data_path('paramsFromQfun.mat', data_dir)
    data = loadmat(path, squeeze_me=True)
    return data['paramsFromQfun']
