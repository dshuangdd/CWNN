from dataclasses import asdict
import json
import hashlib
from pathlib import Path
import time
import os
import tempfile
import numpy as np

from .basis import DesignMatrix, energy_groups, grid_centers, nearest_children
from .config import TrainConfig


class CWNN:


    def __init__(self, dimension, config=None):
        self.config = config or TrainConfig()
        if (
            not isinstance(dimension, (int, np.integer))
            or isinstance(dimension, bool)
            or dimension < 1
        ):
            raise ValueError("dimension must be positive")
        self.dimension = int(dimension)
        centers = grid_centers(
            self.config.initial_resolution,
            self.dimension,
            self.config.translation_range,
            self.config.max_parameters // 2,
        )
        m = self.config.initial_resolution
        self.blocks = [(m, "scaling", centers), (m, "wavelet", centers.copy())]
        self.weights = np.zeros(2 * len(centers))
        self.first_moment = np.zeros_like(self.weights)
        self.second_moment = np.zeros_like(self.weights)
        self.ages = np.zeros(len(self.weights), dtype=np.int64)
        self.history, self.events = [], []
        self.parent_resolution = m
        self.pending_centers = None
        self.pending_groups = None
        self.pending_stage = 0
        self.stage_epochs = 0
        self.has_grown = False
        self.fit_context = None
        self.online_context = None

    @staticmethod
    def _fingerprint(x, y):
        digest = hashlib.sha256()
        for value in (x, y):
            value = np.ascontiguousarray(value, dtype=np.float64)
            digest.update(str(value.shape).encode())
            digest.update(value.tobytes())
        return digest.hexdigest()

    @property
    def parameter_count(self):
        return len(self.weights)

    def _validate_x(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.dimension or len(x) == 0:
            raise ValueError(f"x must have shape (N, {self.dimension}) with N > 0")
        if not np.isfinite(x).all():
            raise ValueError("x contains non-finite values")
        return x

    def _validate_xy(self, x, y):
        x, y = self._validate_x(x), np.asarray(y, dtype=np.float64).reshape(-1)
        if len(x) != len(y) or not np.isfinite(y).all():
            raise ValueError("y must be finite and have one value per input row")
        return x, y

    def predict(self, x):
        return DesignMatrix(self, self._validate_x(x)).predict(self.weights)

    def _level(self, resolution):
        centers, coefficients, offset = [], [], 0
        for m, kind, block_centers in self.blocks:
            end = offset + len(block_centers)
            if m == resolution and kind == "wavelet":
                centers.append(block_centers)
                coefficients.append(self.weights[offset:end])
            offset = end
        return np.concatenate(centers), np.concatenate(coefficients)

    def _append_wavelets(self, resolution, centers):
        existing = set()
        for m, kind, block in self.blocks:
            if m == resolution and kind == "wavelet":
                existing.update(map(tuple, block))
        new = np.array([row for row in centers if tuple(row) not in existing], dtype=np.int64)
        if len(new) == 0:
            return 0
        new = new.reshape(-1, self.dimension)
        if self.parameter_count + len(new) > self.config.max_parameters:
            raise MemoryError("Growth would exceed max_parameters")
        self.blocks.append((resolution, "wavelet", new))
        self.weights = np.pad(self.weights, (0, len(new)))
        self.first_moment = np.pad(self.first_moment, (0, len(new)))
        self.second_moment = np.pad(self.second_moment, (0, len(new)))
        self.ages = np.pad(self.ages, (0, len(new)))
        self.has_grown = True
        self.stage_epochs = 0
        self.events.append(
            {
                "event": "growth",
                "epoch": len(self.history),
                "resolution": resolution,
                "added": len(new),
                "parameters": self.parameter_count,
            }
        )
        return len(new)

    def grow(self):

        while self.parent_resolution < self.config.max_resolution:
            child_m = self.parent_resolution + 1
            if self.config.growth == "full":
                centers = grid_centers(
                    child_m,
                    self.dimension,
                    self.config.translation_range,
                    self.config.max_parameters,
                )
                added = self._append_wavelets(child_m, centers)
                self.parent_resolution = child_m
                return bool(added)
            if self.pending_centers is None:
                self.pending_centers, coefficients = self._level(self.parent_resolution)
                self.pending_groups = energy_groups(coefficients.copy(), self.config.mu)
                self.pending_stage = 0
            last_stage = self.pending_stage == len(self.pending_groups) - 1
            if last_stage:
                centers = grid_centers(
                    child_m,
                    self.dimension,
                    self.config.translation_range,
                    self.config.max_parameters,
                )
            else:
                parent = self.pending_centers[self.pending_groups[self.pending_stage]]
                centers = nearest_children(
                    parent, self.config.translation_range * 2**child_m, self.config.max_parameters
                )
            added = self._append_wavelets(child_m, centers)
            self.pending_stage += 1
            if last_stage:
                self.parent_resolution = child_m
                self.pending_centers = self.pending_groups = None
                self.pending_stage = 0
            if added:
                return True
        return False

    def _update(self, gradient):
        name = self.config.optimizer if self.has_grown else self.config.initial_optimizer
        gradient = self.config.gradient_factor * gradient
        if name == "sgd":
            self.weights -= self.config.learning_rate * gradient
            return
        self.ages += 1
        self.first_moment *= 0.9
        self.first_moment += 0.1 * gradient
        self.second_moment *= 0.999
        self.second_moment += 0.001 * gradient**2
        if name == "adam":
            first = self.first_moment / (1 - 0.9**self.ages)
            second = self.second_moment / (1 - 0.999**self.ages)
            self.weights -= self.config.learning_rate * first / (np.sqrt(second) + 1e-8)
        else:

            self.weights -= (
                self.config.learning_rate * self.first_moment / (np.sqrt(self.second_moment) + 1e-7)
            )

    def fit(self, x, y, x_test=None, y_test=None, phase="train", max_epochs=None):

        x, y = self._validate_xy(x, y)
        if (x_test is None) != (y_test is None):
            raise ValueError("x_test and y_test must be supplied together")
        train = DesignMatrix(self, x)
        test = None
        if x_test is not None:
            x_test, y_test = self._validate_xy(x_test, y_test)
            test = DesignMatrix(self, x_test)
        steps = self.config.max_epochs if max_epochs is None else max_epochs
        if type(steps) is not int or steps < 1:
            raise ValueError("max_epochs must be positive")
        begin = time.perf_counter()
        start_epoch = len(self.history)
        residual = train.predict(self.weights) - y
        previous = float(np.mean(residual**2))
        context = {"fingerprint": self._fingerprint(x, y), "phase": phase}
        if context != self.fit_context:
            self.events.append(
                {
                    "event": "data",
                    "phase": phase,
                    "epoch": start_epoch,
                    "samples": len(x),
                    "mse_before_training": previous,
                }
            )
            self.stage_epochs = 0
        self.fit_context = context
        self.online_context = None
        status = "max_epochs"
        for _ in range(steps):
            if previous <= self.config.target_mse:
                status = "target_reached"
                break
            self._update(train.gradient(residual))
            residual = train.predict(self.weights) - y
            loss = float(np.mean(residual**2))
            if not np.isfinite(loss):
                raise FloatingPointError("Non-finite loss; reduce learning_rate or resolution")
            row = {
                "epoch": len(self.history) + 1,
                "phase": phase,
                "train_mse": loss,
                "parameters": self.parameter_count,
            }
            if test is not None:
                row["test_mse"] = float(np.mean((test.predict(self.weights) - y_test) ** 2))
            self.history.append(row)
            self.stage_epochs += 1
            if loss <= self.config.target_mse:
                status = "target_reached"
                break
            if (
                abs(previous - loss) <= self.config.plateau_tolerance
                and self.stage_epochs >= self.config.min_stage_epochs
            ):
                try:
                    grew = self.grow()
                except MemoryError:
                    status = "parameter_limit"
                    break
                if not grew:
                    status = "resolution_limit"
                    break

            previous = loss
        return {
            "status": status,
            "epochs": len(self.history) - start_epoch,
            "train_mse": float(np.mean((train.predict(self.weights) - y) ** 2)),
            "test_mse": None
            if test is None
            else float(np.mean((test.predict(self.weights) - y_test) ** 2)),
            "parameters": self.parameter_count,
            "seconds": time.perf_counter() - begin,
        }

    def fit_online(
        self,
        x,
        y,
        window=10,
        protocol="disjoint",
        *,
        resume=False,
        stop_on_target=None,
        growth_loss="before_update",
    ):

        x, y = self._validate_xy(x, y)
        if type(window) is not int or window < 1:
            raise ValueError("window must be a positive integer")
        protocol = {"paper_disjoint": "disjoint", "notebook_sliding": "sliding"}.get(
            protocol, protocol
        )
        if protocol not in ("disjoint", "sliding"):
            raise ValueError("Unknown online protocol")
        if growth_loss not in ("before_update", "after_update"):
            raise ValueError("growth_loss must be before_update or after_update")
        begin = time.perf_counter()
        start_epoch = len(self.history)
        status = "stream_complete"
        sliding = protocol == "sliding"
        stop_on_target = sliding if stop_on_target is None else stop_on_target
        if sliding and len(x) < window:
            raise ValueError("A sliding stream needs at least one complete window")
        if resume:
            context = self.online_context
            if (
                context is None
                or context["protocol"] != protocol
                or context["window"] != window
                or context.get("growth_loss") != growth_loss
            ):
                raise ValueError("Resume requires the same saved online protocol/window")
            consumed = context["consumed_stop"]
            if context.get("partial_window") and len(x) > consumed:
                raise ValueError("Cannot extend a stream after its final partial window")
            prefix = np.column_stack((x[:consumed], y[:consumed])).tobytes()
            if len(x) < consumed or hashlib.sha256(prefix).hexdigest() != context["prefix_sha256"]:
                raise ValueError("Previously consumed online samples changed")
            next_start = context["next_start"]
            previous_window_loss = context["previous_window_loss"]
            growth_blocked = context["growth_blocked"]
            growth_status = context.get(
                "growth_status", "blocked_unknown" if growth_blocked else "available"
            )
        else:
            next_start, previous_window_loss, growth_blocked = 0, None, False
            self.stage_epochs = 0
            consumed, prefix = 0, b""
            growth_status = "available"
        prefix_hasher = hashlib.sha256(prefix)
        self.fit_context = None
        starts = (
            range(next_start, len(x) - window + 1) if sliding else range(next_start, len(x), window)
        )
        design = None
        for start in starts:
            if len(self.history) - start_epoch >= self.config.max_epochs:
                status = "max_epochs"
                break
            stop = min(start + window, len(x))
            if sliding and design is not None:
                design.slide(x[stop - 1 : stop])
            else:
                design = DesignMatrix(self, x[start:stop])
            residual = design.predict(self.weights) - y[start:stop]
            pre = float(np.mean(residual**2))
            self._update(design.gradient(residual))
            post = float(np.mean((design.predict(self.weights) - y[start:stop]) ** 2))
            if not np.isfinite(post):
                raise FloatingPointError("Non-finite online loss")
            row = {
                "epoch": len(self.history) + 1,
                "phase": "online",
                "sample_start": start,
                "sample_stop": stop,
                "window_mse_before": pre,
                "train_mse": post,
                "parameters": self.parameter_count,
            }
            monitor = pre if growth_loss == "before_update" else post
            row["growth_monitor_mse"] = monitor
            if not sliding:
                row["prequential_mse"] = pre
            self.history.append(row)
            self.stage_epochs += 1
            prefix_hasher.update(np.column_stack((x[consumed:stop], y[consumed:stop])).tobytes())
            consumed = stop
            self.online_context = dict(
                protocol=protocol,
                window=window,
                next_start=start + (1 if sliding else window),
                consumed_stop=stop,
                previous_window_loss=monitor,
                growth_blocked=growth_blocked,
                growth_loss=growth_loss,
                growth_status=growth_status,
                partial_window=stop - start < window,
                prefix_sha256=prefix_hasher.hexdigest(),
            )
            if stop_on_target and post <= self.config.target_mse:
                status = "target_reached"
                break
            change = (
                abs(previous_window_loss - monitor)
                if previous_window_loss is not None
                else float("inf")
            )
            if (
                not growth_blocked
                and post > self.config.target_mse
                and change <= self.config.plateau_tolerance
                and self.stage_epochs >= self.config.min_stage_epochs
            ):
                try:
                    grew = self.grow()
                except MemoryError:
                    grew, status = False, "parameter_limit"
                    growth_status = "parameter_limit"
                if not grew:
                    growth_blocked = True
                    if growth_status == "available":
                        growth_status = "resolution_limit"
                    if status == "stream_complete":
                        status = "resolution_limit"
            previous_window_loss = monitor
            self.online_context["growth_blocked"] = growth_blocked
            self.online_context["growth_status"] = growth_status
        tail = self.history[start_epoch:][-20:]
        if not tail:
            return {
                "status": "no_new_windows",
                "protocol": protocol,
                "epochs": 0,
                "growth_status": growth_status,
                "total_epochs": len(self.history),
                "parameters": self.parameter_count,
                "last_window_mse": None,
                "last_20_windows_before_mse": None,
                "target_reached_last_window": False,
                "seconds": time.perf_counter() - begin,
            }
        rows = self.history[start_epoch:]
        count = sum(r["sample_stop"] - r["sample_start"] for r in rows)
        weighted_before = (
            sum(r["window_mse_before"] * (r["sample_stop"] - r["sample_start"]) for r in rows)
            / count
        )
        return {
            "status": status,
            "protocol": protocol,
            "epochs": len(self.history) - start_epoch,
            "growth_status": growth_status,
            "total_epochs": len(self.history),
            "parameters": self.parameter_count,
            "last_window_mse": self.history[-1]["train_mse"],
            "last_20_windows_before_mse": float(np.mean([r["window_mse_before"] for r in tail])),
            "target_reached_last_window": self.history[-1]["train_mse"] <= self.config.target_mse,
            "mean_window_mse_before": weighted_before,
            "prequential_mse": None if sliding else weighted_before,
            "stop_on_target": bool(stop_on_target),
            "growth_loss": growth_loss,
            "seconds": time.perf_counter() - begin,
        }

    def save(self, path):

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "format_version": 2,
            "config": asdict(self.config),
            "dimension": self.dimension,
            "blocks": [(m, kind) for m, kind, _ in self.blocks],
            "history": self.history,
            "events": self.events,
            "parent_resolution": self.parent_resolution,
            "pending_stage": self.pending_stage,
            "pending_groups": None
            if self.pending_groups is None
            else [g.tolist() for g in self.pending_groups],
            "stage_epochs": self.stage_epochs,
            "has_grown": self.has_grown,
            "fit_context": self.fit_context,
            "online_context": self.online_context,
        }
        arrays = {f"centers_{i}": block[2] for i, block in enumerate(self.blocks)}
        if path.suffix != ".npz":
            path = Path(str(path) + ".npz")
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=path.parent, suffix=".npz", delete=False
            ) as handle:
                temporary = Path(handle.name)
                np.savez_compressed(
                    handle,
                    metadata=np.array(json.dumps(metadata)),
                    weights=self.weights,
                    first_moment=self.first_moment,
                    second_moment=self.second_moment,
                    ages=self.ages,
                    pending_centers=np.empty((0, self.dimension), dtype=np.int64)
                    if self.pending_centers is None
                    else self.pending_centers,
                    **arrays,
                )
            os.replace(temporary, path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(str(archive["metadata"]))
            if meta["format_version"] not in (1, 2):
                raise ValueError("Unsupported checkpoint format")
            meta["config"].setdefault("basis_family", "spatial-radial")
            model = cls(meta["dimension"], TrainConfig(**meta["config"]))
            model.blocks = [
                (m, kind, archive[f"centers_{i}"].copy())
                for i, (m, kind) in enumerate(meta["blocks"])
            ]
            for name in ("weights", "first_moment", "second_moment", "ages"):
                setattr(model, name, archive[name].copy())
            for name in (
                "history",
                "events",
                "parent_resolution",
                "pending_stage",
                "stage_epochs",
                "has_grown",
            ):
                setattr(model, name, meta[name])
            if meta["pending_groups"] is not None:
                model.pending_groups = [np.array(g, dtype=np.int64) for g in meta["pending_groups"]]
                model.pending_centers = archive["pending_centers"].copy()
            model.fit_context = meta.get("fit_context")
            model.online_context = meta.get("online_context")
            if model.online_context is not None:
                old_protocol = model.online_context["protocol"]
                model.online_context["protocol"] = {
                    "paper_disjoint": "disjoint",
                    "notebook_sliding": "sliding",
                }.get(old_protocol, old_protocol)
            model._validate_checkpoint()
            return model

    def _validate_checkpoint(self):

        for _, kind, centers in self.blocks:
            if (
                kind not in ("scaling", "wavelet")
                or centers.ndim != 2
                or centers.shape[1] != self.dimension
                or centers.dtype.kind not in "iu"
            ):
                raise ValueError("Invalid checkpoint basis block")
        count = sum(len(c) for _, _, c in self.blocks)
        if not 0 < count <= self.config.max_parameters:
            raise ValueError("Invalid checkpoint parameter count")
        for name in ("weights", "first_moment", "second_moment", "ages"):
            value = getattr(self, name)
            allowed = "iu" if name == "ages" else "f"
            if (
                value.shape != (count,)
                or value.dtype.kind not in allowed
                or not np.isfinite(value).all()
            ):
                raise ValueError(f"Invalid checkpoint {name}")
        if (self.second_moment < 0).any() or (self.ages < 0).any():
            raise ValueError("Invalid checkpoint optimizer counters")
