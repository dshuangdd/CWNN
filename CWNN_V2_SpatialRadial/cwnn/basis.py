import itertools
import numpy as np
from scipy.spatial.distance import cdist


def grid_centers(resolution, dimension, translation_range, max_centers=100000):

    if (
        type(resolution) is not int
        or resolution < 0
        or type(dimension) is not int
        or dimension < 1
        or type(translation_range) is not int
        or translation_range < 1
        or max_centers < 1
    ):
        raise ValueError("Invalid grid dimensions or budget")
    side = translation_range * 2**resolution + 1
    count = side**dimension
    if count > max_centers:
        raise MemoryError(f"Grid needs {count:,} centers; limit is {max_centers:,}")
    ids = np.arange(count, dtype=np.int64)
    return np.stack(np.unravel_index(ids, (side,) * dimension), axis=1)


def nearest_children(parents, upper, max_centers=100000):

    raw = np.asarray(parents)
    if (
        raw.ndim != 2
        or raw.shape[1] < 1
        or not np.isfinite(raw).all()
        or not np.equal(raw, np.floor(raw)).all()
    ):
        raise ValueError("parents must be a finite integer-coordinate matrix")
    if not isinstance(upper, (int, np.integer)) or upper < 1 or max_centers < 1:
        raise ValueError("upper and max_centers must be positive integers")
    parents = raw.astype(np.int64)
    if (parents < 0).any() or (2 * parents > upper).any():
        raise ValueError("Parents lie outside the next-resolution domain")
    dimension = parents.shape[1]
    if len(parents) == 0:
        return np.empty((0, dimension), dtype=np.int64)
    if 2**dimension > max_centers:
        raise MemoryError("A single parent exceeds the child-center budget")
    bits = np.array(list(itertools.product((0, 1), repeat=dimension)), dtype=np.int64)


    side = upper + 1
    encoded = side**dimension <= np.iinfo(np.int64).max
    powers = (
        np.array([side**p for p in range(dimension - 1, -1, -1)], dtype=np.int64)
        if encoded
        else None
    )
    found = np.empty(0, dtype=np.int64) if encoded else np.empty((0, dimension), dtype=np.int64)
    batch = max(1, 16384 // len(bits))
    for start in range(0, len(parents), batch):
        exact = 2 * parents[start : start + batch, None, :]
        adjacent = np.where(exact == 0, 1, exact - 1)
        children = np.where(bits[None] == 0, exact, adjacent).reshape(-1, dimension)
        if encoded:
            found = np.union1d(found, children @ powers)
        else:
            found = np.unique(np.concatenate((found, children)), axis=0)
        if len(found) > max_centers:
            raise MemoryError("Selected children exceed the center budget")
    return (found[:, None] // powers % side) if encoded else found


def energy_groups(weights, mu):

    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1 or not np.isfinite(weights).all():
        raise ValueError("Energy weights must be a finite vector")
    if not np.isfinite(mu) or not 0 < mu <= 1 or not np.isclose(1 / mu, round(1 / mu)):
        raise ValueError("mu must be the reciprocal of a positive integer")
    groups = round(1 / mu)
    if len(weights) == 0:
        return [np.empty(0, dtype=np.int64) for _ in range(groups)]

    largest = np.abs(weights).max()
    energies = (weights / largest) ** 2 if largest else np.zeros_like(weights)
    order = np.argsort(-energies, kind="stable")
    cumulative = np.cumsum(energies[order])
    result, start = [], 0
    for stage in range(1, groups + 1):
        if stage == groups:
            end = len(order)
        elif cumulative[-1] == 0:
            end = int(np.ceil(len(order) * stage / groups))
        else:
            end = int(np.searchsorted(cumulative, stage / groups * cumulative[-1], side="left")) + 1
        result.append(order[start:end])
        start = end
    return result


def radial_basis(x, centers, resolution, kind="wavelet", normalization="notebook"):

    x, centers = np.asarray(x, dtype=np.float64), np.asarray(centers, dtype=np.float64)
    if (
        x.ndim != 2
        or centers.ndim != 2
        or x.shape[1] != centers.shape[1]
        or not np.isfinite(x).all()
        or not np.isfinite(centers).all()
        or type(resolution) is not int
        or resolution < 0
    ):
        raise ValueError("Invalid basis inputs or resolution")
    if normalization not in ("notebook", "l2"):
        raise ValueError("Unknown basis normalization")
    distance = cdist(x * 2**resolution, centers, metric="euclidean")
    phi = np.sinc(distance / np.pi)
    if kind == "wavelet":
        phi -= 2 * np.sinc(2 * distance / np.pi)
    elif kind != "scaling":
        raise ValueError("kind must be scaling or wavelet")
    gain = 2.0 ** (resolution if normalization == "notebook" else resolution * x.shape[1] / 2)
    return phi * gain


class DesignMatrix:


    def __init__(self, model, x, cache_mb=None):
        self.model = model
        self.x = np.array(x, dtype=np.float64, copy=True)
        self.budget = (model.config.cache_mb if cache_mb is None else cache_mb) * 1024**2
        self.matrix = None
        self.cached_blocks = 0

    def blocks(self, start=0, stop=None, x=None):
        x = self.x if x is None else x
        offset = sum(len(block[2]) for block in self.model.blocks[:start])
        for resolution, kind, centers in self.model.blocks[start:stop]:
            for first in range(0, len(centers), self.model.config.chunk_size):
                part = centers[first : first + self.model.config.chunk_size]
                yield slice(offset + first, offset + first + len(part)), radial_basis(
                    x, part, resolution, kind, self.model.config.normalization
                )
            offset += len(centers)

    def slide(self, new_rows):

        count = len(new_rows)
        self.x = np.concatenate((self.x[count:], new_rows))
        if self.matrix is None:
            return
        out = np.empty(self.matrix.shape, dtype=np.float64, order="F")
        out[:-count] = self.matrix[count:]
        for columns, values in self.blocks(stop=self.cached_blocks, x=new_rows):
            out[-count:, columns] = values
        self.matrix = out

    def refresh(self):
        n, p = len(self.x), self.model.parameter_count
        if n * p * 8 > self.budget:
            self.matrix, self.cached_blocks = None, 0
            return
        if self.matrix is not None and self.cached_blocks == len(self.model.blocks):
            return
        out = np.empty((n, p), dtype=np.float64, order="F")
        if self.matrix is not None:
            out[:, : self.matrix.shape[1]] = self.matrix
        for columns, values in self.blocks(self.cached_blocks):
            out[:, columns] = values
        self.matrix, self.cached_blocks = out, len(self.model.blocks)

    def predict(self, weights):
        self.refresh()
        if self.matrix is not None:
            return self.matrix @ weights
        out = np.zeros(len(self.x))
        for columns, values in self.blocks():
            out += values @ weights[columns]
        return out

    def gradient(self, residual):
        self.refresh()
        if self.matrix is not None:
            return self.matrix.T @ residual / len(self.x)
        out = np.empty(self.model.parameter_count)
        for columns, values in self.blocks():
            out[columns] = values.T @ residual / len(self.x)
        return out
