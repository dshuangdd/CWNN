from dataclasses import dataclass
import numpy as np


def input_matrix(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or min(x.shape) < 1 or not np.isfinite(x).all():
        raise ValueError("Inputs must be a finite, nonempty (samples, dimensions) matrix")
    return x


class InputScaler:


    def fit(self, x):
        x = input_matrix(x)
        self.minimum = x.min(axis=0)
        self.span = x.max(axis=0) - self.minimum
        self.span[self.span == 0] = 1.0
        return self

    def transform(self, x):
        x = input_matrix(x)
        if x.shape[1] != len(self.minimum):
            raise ValueError("Input dimension differs from the fitted scaler")
        return (x - self.minimum) / self.span

    def fit_transform(self, x):
        return self.fit(x).transform(x)

    def save(self, path):
        np.savez_compressed(path, minimum=self.minimum, span=self.span)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as saved:
            scaler = cls()
            scaler.minimum, scaler.span = saved["minimum"].copy(), saved["span"].copy()
        return scaler


@dataclass
class Dataset:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray | None = None
    y_test: np.ndarray | None = None


def load_dataset(path):

    with np.load(path, allow_pickle=False) as saved:
        x_train, y_train = _pair(saved["x_train"], saved["y_train"])
        if ("x_test" in saved) != ("y_test" in saved):
            raise ValueError("x_test and y_test must be supplied together")
        x_test, y_test = (
            _pair(saved["x_test"], saved["y_test"]) if "x_test" in saved else (None, None)
        )
    if x_test is not None and x_test.shape[1] != x_train.shape[1]:
        raise ValueError("Training and test inputs must have the same number of columns")
    return Dataset(x_train, y_train, x_test, y_test)


def _pair(x, y):
    x = input_matrix(x)
    y = np.asarray(y, dtype=np.float64)
    if y.shape == (len(x), 1):
        y = y[:, 0]
    if y.shape != (len(x),) or not np.isfinite(y).all():
        raise ValueError("Targets must contain one finite scalar per input row")
    return x, y
