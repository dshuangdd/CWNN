from functools import lru_cache
import math
import operator

import numpy as np
from scipy.special import j0, j1, jv


def _dimension(value):
    try:
        dimension = operator.index(value)
    except TypeError as error:
        raise ValueError("dimension must be a positive integer") from error
    if isinstance(value, (bool, np.bool_)) or dimension < 1:
        raise ValueError("dimension must be a positive integer")
    return dimension


def _squared_radii(r2):
    values = np.asarray(r2, dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("r2 must contain finite non-negative squared radii")
    return values


def _check_band(omega_low, omega_high):
    if not (math.isfinite(omega_low) and math.isfinite(omega_high)
            and 0 < omega_low < omega_high):
        raise ValueError("require finite 0 < omega_low < omega_high")


@lru_cache(maxsize=32)
def _series(nu):
    coefficients = np.array([
        (-1.)**k / (4.**k * math.factorial(k) * 2.**nu * math.gamma(nu+k+1))
        for k in range(13)
    ], dtype=np.float64)
    coefficients.setflags(write=False)
    return coefficients


def _ratios(z, nu, derivative, backend):
    value = np.empty_like(z)
    adjacent = np.empty_like(z) if derivative else None
    near = z <= 2.
    if np.any(near):
        squared = z[near]**2
        value[near] = np.polynomial.polynomial.polyval(squared, _series(nu))
        if derivative:
            adjacent[near] = np.polynomial.polynomial.polyval(squared, _series(nu+1))
    far = z[~near]
    if far.size:
        if backend == "fast" and nu in (1., 2., 3., 4.):
            previous, current = j0(far), j1(far)
            for order in range(1, int(nu)):
                previous, current = current, (2*order/far)*current-previous
            value[~near] = current/far**nu
            if derivative:
                adjacent[~near] = ((2*nu/far)*current-previous)/far**(nu+1)
        elif backend == "fast" and nu in (.5, 1.5, 2.5, 3.5, 4.5):
            order = int(nu - .5)
            reciprocal = 1 / far
            reciprocal2 = reciprocal * reciprocal
            previous = np.sin(far) * reciprocal
            current = (previous - np.cos(far)) * reciprocal2
            factor = math.sqrt(2 / np.pi)
            if order == 0:
                value[~near] = factor * previous
                if derivative:
                    adjacent[~near] = factor * current
            else:
                for n in range(1, order):
                    previous, current = current, ((2 * n + 1) * current - previous) * reciprocal2
                value[~near] = factor * current
                if derivative:
                    adjacent[~near] = factor * ((2 * order + 1) * current - previous) * reciprocal2
        else:
            value[~near] = jv(nu, far)/far**nu
            if derivative:
                adjacent[~near] = jv(nu+1, far)/far**(nu+1)
    return value, adjacent


def _ball(r, dimension, radius, derivative, backend):
    nu = dimension / 2
    ratio, adjacent = _ratios(radius*r, nu, derivative, backend)
    scale = radius**dimension / (2*np.pi)**nu
    return scale*ratio, -scale*radius**2*adjacent if derivative else None


def ball_kernel(r2, dimension, radius=1., *, derivative=False, backend="fast"):
    dimension = _dimension(dimension)
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be finite and positive")
    if backend not in {"fast", "reference"}:
        raise ValueError("backend must be fast or reference")
    return _ball(np.sqrt(_squared_radii(r2)), dimension, radius, derivative, backend)


def radial_kernels(r2, dimension, *, omega_low=1., omega_high=2.,
                   derivative=False, backend="fast"):
    dimension = _dimension(dimension)
    _check_band(omega_low, omega_high)
    if backend not in {"fast", "reference"}:
        raise ValueError("backend must be fast or reference")
    r = np.sqrt(_squared_radii(r2))
    low, glow = _ball(r, dimension, omega_low, derivative, backend)
    high, ghigh = _ball(r, dimension, omega_high, derivative, backend)
    return low, high-low, glow, ghigh-glow if derivative else None


def mother_norms2(dimension, *, omega_low=1., omega_high=2.):
    dimension = _dimension(dimension)
    _check_band(omega_low, omega_high)
    factor = 1 / ((4*np.pi)**(dimension/2) * math.gamma(dimension/2+1))
    return factor*omega_low**dimension, factor*(omega_high**dimension-omega_low**dimension)
