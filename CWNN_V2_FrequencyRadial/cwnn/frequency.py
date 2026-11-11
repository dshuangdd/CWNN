import math
import numpy as np
from .basis import grid_centers, nearest_children, radial_basis


def estimate_frequency(
    x,
    y,
    centers,
    start_resolution=1,
    learning_rate=0.005,
    epsilon=0.006,
    probe_steps=10,
    max_resolution=6,
    max_centers=350000,
    normalization="l2",
    optimizer="adam",
    translation_range=None,
    chunk_size=2048,
    cache_mb=256,
    basis_family="frequency-radial",
):
    x = np.asarray(x, dtype=float)
    if (
        not 0 < epsilon < 1
        or type(probe_steps) is not int
        or probe_steps < 1
        or not math.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ValueError("epsilon must be in (0, 1); steps and rate must be positive")
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float).reshape(-1)
    raw_centers = np.asarray(centers)
    if not np.isfinite(raw_centers).all() or not np.equal(raw_centers, np.floor(raw_centers)).all():
        raise ValueError("Probe centers must be finite integer coordinates")
    centers = raw_centers.astype(np.int64)
    if x.ndim != 2 or len(x) == 0 or len(x) != len(y):
        raise ValueError("x must be nonempty 2D data with one target per row")
    if centers.ndim != 2 or centers.shape[1] != x.shape[1] or len(centers) == 0:
        raise ValueError("centers must be nonempty and match the input dimension")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Frequency inputs and targets must be finite")
    if optimizer not in ("sgd", "adam", "legacy_adam"):
        raise ValueError("Probe optimizer must be sgd, adam, or legacy_adam")
    settings = {"probe_steps": probe_steps, "optimizer": optimizer,
                "basis_family": basis_family, "normalization": normalization}
    alpha = 2 * math.atan(-math.log10(epsilon)) / math.pi
    ema, trace = None, []
    if (
        type(start_resolution) is not int
        or start_resolution < 1
        or type(max_resolution) is not int
        or max_resolution < start_resolution
        or type(max_centers) is not int
        or max_centers < 1
        or type(chunk_size) is not int
        or chunk_size < 1
    ):
        raise ValueError("Invalid probe resolution or memory budget")
    upper_physical = (
        math.ceil(int(np.max(centers)) / 2**start_resolution)
        if translation_range is None
        else translation_range
    )
    if (
        upper_physical < 1
        or (centers < 0).any()
        or (centers > upper_physical * 2**start_resolution).any()
    ):
        raise ValueError("Probe centers are outside the translation domain")
    for count, resolution in enumerate(range(start_resolution, max_resolution + 1), 1):
        if len(centers) > max_centers:
            return {
                **settings,
                "status": "center_limit",
                "initial_resolution": None,
                "trace": trace,
            }
        weights = np.zeros(len(centers))
        first, second = np.zeros_like(weights), np.zeros_like(weights)

        cached = None
        if len(x) * len(centers) * 8 <= cache_mb * 1024**2:
            cached = np.empty((len(x), len(centers)))
            for start in range(0, len(centers), chunk_size):
                cached[:, start : start + chunk_size] = radial_basis(
                    x, centers[start : start + chunk_size], resolution,
                    normalization=normalization, basis_family=basis_family,
                )
        for step in range(probe_steps):
            prediction = np.zeros(len(x)) if cached is None else cached @ weights
            if step and cached is None:
                for start in range(0, len(centers), chunk_size):
                    values = radial_basis(
                        x,
                        centers[start : start + chunk_size],
                        resolution,
                        normalization=normalization,
                        basis_family=basis_family,
                    )
                    prediction += values @ weights[start : start + chunk_size]
            if cached is not None:
                gradient = cached.T @ (prediction - y) / len(x)
            else:
                gradient = np.empty_like(weights)
                for start in range(0, len(centers), chunk_size):
                    values = radial_basis(
                        x,
                        centers[start : start + chunk_size],
                        resolution,
                        normalization=normalization,
                        basis_family=basis_family,
                    )
                    gradient[start : start + chunk_size] = values.T @ (prediction - y) / len(x)
            if optimizer == "sgd":

                weights -= learning_rate * 2 * gradient
            elif optimizer == "legacy_adam":

                first = 0.9 * first + 0.1 * gradient
                second = 0.999 * second + 0.001 * gradient**2
                weights -= learning_rate * first / (np.sqrt(second) + 1e-7)
            else:
                gradient *= 2
                first = 0.9 * first + 0.1 * gradient
                second = 0.999 * second + 0.001 * gradient**2
                corrected_first = first / (1 - 0.9 ** (step + 1))
                corrected_second = second / (1 - 0.999 ** (step + 1))
                weights -= learning_rate * corrected_first / (np.sqrt(corrected_second) + 1e-8)
        energy = float(weights @ weights)
        if not math.isfinite(energy):
            raise FloatingPointError("Non-finite probe energy")
        previous = ema

        ema = energy if ema is None else (alpha * ema + (1 - alpha) * energy) / (1 - alpha**count)
        trace.append(
            {"resolution": resolution, "centers": len(centers), "energy": energy, "ema": ema}
        )
        if previous is not None and energy <= previous:


            return {
                **settings,
                "status": "energy_drop",
                "initial_resolution": resolution - 2,
                "trace": trace,
            }
        if resolution < max_resolution:
            try:
                centers = nearest_children(
                    centers, upper_physical * 2 ** (resolution + 1), max_centers
                )
            except MemoryError:
                return {
                    **settings,
                    "status": "center_limit",
                    "initial_resolution": None,
                    "trace": trace,
                }
    return {**settings, "status": "resolution_limit", "initial_resolution": None, "trace": trace}


def estimate_resolution(
    x,
    y,
    *,
    learning_rate=0.005,
    target_mse=0.006,
    probe_steps=10,
    max_resolution=5,
    max_centers=100000,
    translation_range=1,
    cache_mb=256,
    chunk_size=2048,
    normalization="l2",
    basis_family="frequency-radial",
):

    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or min(x.shape) < 1:
        raise ValueError("x must be a nonempty matrix")
    try:
        centers = 2 * grid_centers(0, x.shape[1], translation_range, max_centers)
    except MemoryError:
        return dict(
            status="center_limit",
            initial_resolution=None,
            trace=[],
            probe_steps=probe_steps,
            optimizer="adam",
            dimension=x.shape[1],
            basis_family=basis_family,
            normalization=normalization,
        )
    result = estimate_frequency(
        x,
        y,
        centers,
        learning_rate=learning_rate,
        epsilon=target_mse,
        probe_steps=probe_steps,
        max_resolution=max_resolution,
        max_centers=max_centers,
        translation_range=translation_range,
        cache_mb=cache_mb,
        chunk_size=chunk_size,
        normalization=normalization,
        basis_family=basis_family,
    )
    return {"dimension": x.shape[1], "samples": len(x), **result}
