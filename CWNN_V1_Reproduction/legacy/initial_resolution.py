from itertools import product
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits


def _wavelets_2d(data, resolution, centers):
    centers = centers.reshape(len(centers), 1, 2)
    radius = np.sum((2**resolution * data - centers) ** 2, axis=2) ** 0.5
    radius = radius + np.pi / 1_000_000.0
    return 2**resolution * (-np.sin(2 * radius) / radius + np.sin(radius) / radius)


def _children_2d(centers):
    children = dict.fromkeys(
        (first, second)
        for parent in centers
        for first in ((0.0, 1.0) if parent[0] == 0 else (2 * parent[0] - 1, 2 * parent[0]))
        for second in ((0.0, 1.0) if parent[1] == 0 else (2 * parent[1] - 1, 2 * parent[1]))
    )
    return np.asarray(list(children), dtype=float)


def _probe_2d(data, targets, epsilon, online=False, max_resolution=8):
    centers = np.asarray(list(product((0, 2, 4), repeat=2)))
    alpha = 2 * np.arctan(-np.log10(epsilon)) / np.pi
    smoothed = []
    trace = []
    for resolution in range(1, max_resolution + 1):
        weights = np.zeros(len(centers))
        first_moment = np.zeros_like(weights)
        second_moment = np.zeros_like(weights)
        fixed_wavelets = None if online else _wavelets_2d(data, resolution, centers)
        for step in range(1, 11):
            active_targets = targets[:step] if online else targets
            active_data = data[:, :step, :] if online else data
            wavelets = (
                _wavelets_2d(active_data, resolution, centers)
                if online else fixed_wavelets
            )
            predictions = np.dot(wavelets.T, weights)
            divisor = 1 if online else data.shape[1]
            gradient = np.dot(wavelets, predictions - active_targets) / divisor
            first_moment = 0.9 * first_moment + 0.1 * gradient
            second_moment = 0.999 * second_moment + 0.001 * gradient**2
            weights = weights - 0.0005 * first_moment / (np.sqrt(second_moment) + 1e-7)

        energy = float(sum(weights**2))
        previous = smoothed[-1] if smoothed else None
        estimate = energy
        if previous is not None:
            estimate = float(
                (alpha * previous + (1 - alpha) * energy)
                / (1 - alpha ** (len(smoothed) + 1))
            )
        smoothed.append(estimate)
        stopped = previous is not None and energy <= previous
        trace.append(dict(
            resolution=resolution,
            centers=len(centers),
            energy=energy,
            smoothed_energy=estimate,
            stop_on_energy=stopped,
        ))
        if stopped:
            return dict(
                status="energy_stop", initial_resolution=resolution - 2,
                samples=data.shape[1], probe_updates_per_resolution=10,
                epsilon=epsilon, trace=trace,
            )
        if resolution < max_resolution:
            centers = _children_2d(centers)
    return dict(
        status="resolution_budget_exhausted", initial_resolution=None,
        samples=data.shape[1], probe_updates_per_resolution=10,
        epsilon=epsilon, trace=trace,
    )


def _mapping_data(noise_scale):
    np.random.seed(1234)
    first = np.arange(100) / 100
    second = np.sqrt(first)
    noise = np.random.normal(0, noise_scale * (1 - first**2), len(first))
    inputs = np.column_stack((first, second))[None, :, :]
    targets = -1 + first + second + np.sin(2 * np.pi * (first + second)) + noise
    return inputs, targets


def _online_data(changed):
    length = 1000 if changed else 500
    np.random.seed(1234)
    values = [1, 1]
    for index in range(2, length):
        argument = np.pi * (values[index - 1] ** 2 + values[index - 2] ** 2)
        value = np.arctan(argument) ** 0.5
        if changed and index >= 9:
            value += np.cos(argument)
        values.append(value)
    noisy = np.asarray(values) + np.random.normal(0, 0.01, length)
    inputs = np.column_stack((noisy[:-2], noisy[1:-1]))[None, :, :]
    targets = noisy[2:]
    return inputs, targets


def _milling_data(test_size, data_path):
    import torch
    from milling_data import load_milling

    data = load_milling(data_path)
    return tuple(torch.tensor(data[key], dtype=torch.float32)
                 for key in ("x_train", "y_train", "x_test", "y_test"))


def _milling_centers(resolution):

    import torch

    if resolution == 1:
        axis = (0.0, 2.0)
    else:

        axis = (0.0, 3.0, 4.0)
    return torch.tensor(list(product(axis, repeat=9)), dtype=torch.float32)


def _milling_wavelets(data, resolution, centers):
    import torch

    radius = torch.norm(2**resolution * data.unsqueeze(1) - centers, p=2, dim=2)
    radius = radius + 1e-6
    return 2**resolution * (-torch.sin(2 * radius) / radius + torch.sin(radius) / radius)


def _probe_milling(test_size, data_path):
    import torch

    torch.manual_seed(1234)
    x_train, y_train, x_test, y_test = _milling_data(test_size, data_path)
    weights = torch.nn.ParameterList()
    trace = []
    for resolution in (1, 2):
        centers = _milling_centers(resolution)
        wavelets_train = torch.cat([
            _milling_wavelets(x_train, resolution, centers[start:start+512])
            for start in range(0, len(centers), 512)
        ], dim=1)
        weights.append(torch.nn.Parameter(torch.zeros(len(centers))))
        optimizer = torch.optim.AdamW([weights[-1]], lr=0.001)
        for _ in range(30):
            optimizer.zero_grad()
            prediction = wavelets_train @ weights[-1]
            train_loss = torch.nn.functional.mse_loss(prediction, y_train)
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(weights.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                current_loss = torch.nn.functional.mse_loss(
                    wavelets_train @ weights[-1], y_train
                )
        trace.append(dict(
            resolution=resolution,
            centers=len(centers),
            energy=float(torch.sum(weights[-1] ** 2).item()),
            last_pre_update_train_mse=float(train_loss.item()),
            last_post_update_train_mse=float(current_loss.item()),
        ))
    stopped = trace[1]["energy"] <= trace[0]["energy"]
    return dict(
        status="energy_stop" if stopped else "two_layer_probe_inconclusive",
        initial_resolution=0 if stopped else None,
        resolution_inferred_from_first_comparison=True,
        train_samples=len(x_train),
        probe_updates_per_resolution=30,
        trace=trace,
    )


def run_example(example, milling_data=None):

    with threadpool_limits(limits=1):
        if example == 1:
            return {
                label: _probe_2d(*_mapping_data(noise), epsilon)
                for label, noise, epsilon in (
                    ("D1", 0.0, 0.005),
                    ("D2", 0.1, 0.005),
                    ("D3", 0.2, 0.025),
                )
            }
        if example == 2:
            data, targets = _mapping_data(0.0)
            return {"DS1": _probe_2d(data[:, :60, :], targets[:60], 0.005)}
        if example == 3:
            return {
                "stationary": _probe_2d(*_online_data(False), 0.025, online=True),
                "changed_initial_window": _probe_2d(*_online_data(True), 0.025, online=True),
            }
        if example == 4:
            data_path = milling_data or Path(__file__).resolve().parents[1] / "data/milling_features.npz"
            return {
                "milling": _probe_milling(0.2, data_path),
            }
        raise ValueError("Example must be 1, 2, 3, or 4")
