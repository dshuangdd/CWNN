from dataclasses import dataclass
import math


@dataclass(frozen=True)
class TrainConfig:
    initial_resolution: int = 0
    translation_range: int = 1
    learning_rate: float = 0.0005
    target_mse: float = 0.006
    plateau_tolerance: float = 0.00004
    mu: float = 1 / 3
    max_epochs: int = 10000
    max_resolution: int = 6
    max_parameters: int = 100000
    min_stage_epochs: int = 2
    optimizer: str = "adam"
    initial_optimizer: str = "adam"
    gradient_factor: float = 2.0
    normalization: str = "notebook"
    cache_mb: float = 256.0
    chunk_size: int = 2048
    growth: str = "energy"

    def __post_init__(self):
        for key in ("learning_rate", "target_mse", "plateau_tolerance", "mu", "gradient_factor"):
            value = getattr(self, key)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be positive and finite")
        if self.mu > 1 or not math.isclose(1 / self.mu, round(1 / self.mu)):
            raise ValueError("mu must be the reciprocal of a positive integer")
        for key in (
            "max_epochs",
            "max_parameters",
            "min_stage_epochs",
            "chunk_size",
            "translation_range",
        ):
            value = getattr(self, key)
            if type(value) is not int or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if type(self.initial_resolution) is not int or self.initial_resolution < 0:
            raise ValueError("initial_resolution must be a nonnegative integer")
        if type(self.max_resolution) is not int or self.max_resolution < self.initial_resolution:
            raise ValueError("max_resolution must include the initial resolution")
        if not math.isfinite(self.cache_mb) or self.cache_mb < 0:
            raise ValueError("cache_mb must be finite and nonnegative")
        if self.normalization not in ("notebook", "l2"):
            raise ValueError("normalization must be notebook or l2")
        if self.growth not in ("energy", "full"):
            raise ValueError("growth must be energy or full")
        for name in (self.optimizer, self.initial_optimizer):
            if name not in ("sgd", "adam", "legacy_adam"):
                raise ValueError("optimizer must be sgd, adam or legacy_adam")
