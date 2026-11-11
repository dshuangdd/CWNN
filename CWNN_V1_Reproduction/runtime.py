from collections import OrderedDict
import functools
import inspect


class Accelerator:
    def __init__(self, enabled=True, cache_mb=256):
        self.enabled = enabled
        self.budget = int(cache_mb * 1024**2)
        self.cache = OrderedDict()
        self.used = 0
        self.hits = self.misses = 0
        self.training_calls = []

    def cached(self, key, compute, retained=()):
        if not self.enabled:
            return compute()
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            return self.cache[key][0]
        self.misses += 1
        value = compute()
        arrays = value if isinstance(value, tuple) else (value,)
        size = sum(
            v.nbytes if hasattr(v, "nbytes") else v.numel() * v.element_size() for v in arrays
        )
        size += sum(len(k) for k in key if isinstance(k, bytes))
        if size <= self.budget:
            while self.cache and self.used + size > self.budget:
                _, (_, old_size, _) = self.cache.popitem(last=False)
                self.used -= old_size
            self.cache[key] = (value, size, retained)
            self.used += size
        return value

    def numpy_class(self, cls):
        original = cls.sinc

        @functools.wraps(original)
        def sinc(model, x, j, k):
            if not self.enabled:
                return original(model, x, j, k)
            key = ("numpy", j, x.shape, x.dtype.str, x.tobytes(), k.shape, k.dtype.str, k.tobytes())
            return self.cached(key, lambda: original(model, x, j, k))

        cls.sinc = sinc
        train = getattr(cls, "train", None)
        if train is None:
            return cls
        signature = inspect.signature(train)
        if "pre_ac" in signature.parameters:

            @functools.wraps(train)
            def tracked_train(model, *args, **kwargs):
                bound = signature.bind(model, *args, **kwargs)
                result = train(model, *args, **kwargs)

                losses = result[-3] if "data_test" in signature.parameters else result[-1]
                if "data_test" in signature.parameters:
                    end_epoch = result[1] if len(result) == 5 else result[2]
                    updates = end_epoch - bound.arguments.get("epoch_last", 0)
                else:
                    updates = len(losses)
                self.training_calls.append(
                    dict(
                        samples=bound.arguments["data"].shape[1],
                        updates=updates,
                        parameters=len(model.w1) + (len(model.w2) if hasattr(model, "w2") else 0),
                        last_mse_before_update=float(losses[-1]) if len(losses) else None,
                    )
                )
                return result

            cls.train = tracked_train
        return cls

    def torch_class(self, cls):
        if not self.enabled:
            return cls
        import torch

        original = cls.forward

        def forward(model, x):

            if x.requires_grad:
                return original(model, x)
            if x.device != model.device:
                x = x.to(model.device)
            output = 0
            for index, weights in enumerate(model.weights):
                active = model.active_indices[index]
                centers = model.centers_list[index][active]
                j = model.j + index
                key = (
                    "torch",
                    x.data_ptr(),
                    x._version,
                    tuple(x.shape),
                    tuple(x.stride()),
                    str(x.dtype),
                    str(x.device),
                    j,
                    centers.cpu().numpy().tobytes(),
                )

                def compute():

                    y = torch.norm(2**j * x.unsqueeze(1) - centers, p=2, dim=2) + 1e-6
                    phi = 2**j * torch.sin(y) / y
                    psi = 2**j * (-torch.sin(2 * y) / y + torch.sin(y) / y)
                    return phi, psi

                phi, psi = self.cached(key, compute, (x,))
                if index == 0:
                    output += phi @ weights[: len(active)] + psi @ weights[len(active) :]
                else:
                    output += psi @ weights[: len(active)]
            return output

        cls.forward = forward
        return cls
