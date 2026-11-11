import time

import numpy as np
import torch
from torch import nn
from threadpoolctl import threadpool_limits


class ResidualBlock(nn.Module):


    def __init__(self, hidden_size):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
        )
        for layer in self.block:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")

    def forward(self, x):
        return torch.relu(x + self.block(x))


class GNN(nn.Module):


    def __init__(self, input_size, num_blocks):
        super().__init__()
        self.input_layer = nn.Sequential(nn.Linear(input_size, 128), nn.BatchNorm1d(128), nn.ReLU())
        self.res_blocks = nn.Sequential(*[ResidualBlock(128) for _ in range(num_blocks)])
        self.output_layer = nn.Linear(128, 1)

    def forward(self, x):
        return self.output_layer(self.res_blocks(self.input_layer(x)))


def run(experiment, depth, data_path=None, max_epochs=10000, seed=1234,
        profile="archive", offset=None):

    if experiment not in ("D1", "D2", "D3", "milling") or depth not in (3, 5, 7):
        raise ValueError("Choose D1/D2/D3/milling and depth 3/5/7")
    if max_epochs < 1:
        raise ValueError("max_epochs must be positive")
    if profile not in ("archive", "paper"):
        raise ValueError("profile must be archive or paper")
    np.random.seed(seed)
    torch.manual_seed(seed)
    begin = time.perf_counter()
    if experiment == "milling":
        from milling_data import load_milling

        prepared = load_milling(data_path, seed)
        x = np.vstack((prepared["x_train"], prepared["x_test"]))
        y = np.concatenate((prepared["y_train"], prepared["y_test"]))
        target, budget = 0.015, min(max_epochs, 3000)
    else:
        if offset is None:
            offset = 0.5 if profile == "paper" else -1.0
        x1 = np.arange(100) / 100
        x2 = np.sqrt(x1)
        x = np.hstack((x1.reshape(-1, 1), x2.reshape(-1, 1)))
        noise = np.random.normal(0, {"D1": 0, "D2": 0.1, "D3": 0.2}[experiment] * (1 - x1**2))
        y = offset + (x1 + x2) + np.sin(2 * np.pi * (x1 + x2)) + noise
        target = 0.025 if experiment == "D3" else 0.006
        budget = min(max_epochs, 2000 if depth == 3 else 1300)
    x, y = torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).view(-1, 1)
    if experiment == "milling":
        train_indices = np.arange(len(prepared["y_train"]))
        test_indices = np.arange(len(prepared["y_train"]), len(y))
    else:

        indices = np.random.RandomState(seed).permutation(len(x))
        train_size = int(0.8 * len(x))
        train_indices, test_indices = indices[:train_size], indices[train_size:]
    x_train, x_test = x[train_indices], x[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    mean = torch.mean(x_train, dim=0, keepdim=True)
    std = torch.std(x_train, dim=0, keepdim=True, unbiased=False)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    normalized_train, normalized_test = (x_train - mean) / std, (x_test - mean) / std
    num_blocks = (depth - 3) // 2
    model = GNN(x.shape[1], num_blocks)
    losses, test_losses, after_losses = [], [], []
    with threadpool_limits(1):
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        criterion = nn.MSELoss()
        for _ in range(budget):
            model.train()
            optimizer.zero_grad()
            loss = criterion(model(normalized_train), y_train)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            model.eval()
            with torch.no_grad():
                test_losses.append(criterion(model(normalized_test), y_test).item())
                if experiment == "milling":
                    after_losses.append(criterion(model(normalized_train), y_train).item())
            stopping_loss = after_losses[-1] if experiment == "milling" else losses[-1]
            if stopping_loss <= target:
                break
        with torch.no_grad():
            final_train = criterion(model(normalized_train), y_train).item()
    weights = np.concatenate([p.detach().numpy().ravel() for p in model.parameters()])
    source_stop = (after_losses[-1] if experiment == "milling" else losses[-1]) <= target
    summary = dict(
        experiment=experiment,
        method=f"GNN-{depth}",
        profile=profile,
        seed=seed,
        device="cpu",
        status=("target_reached" if final_train <= target else "archived_stop_only")
        if source_stop
        else "budget_limit",
        epochs=len(losses),
        parameters=len(weights),
        target=target,
        final_model_target_reached=final_train <= target,
        last_train_mse_before_update=losses[-1],
        last_train_mse_after_update=final_train,
        last_test_mse_after_update=test_losses[-1],
        seconds=time.perf_counter() - begin,
        stopping_loss="train_mse_after_update" if experiment == "milling" else "train_mse_before_update_in_train_mode",
        final_train_mode="eval",
        settings=dict(
            hidden_size=128, num_blocks=num_blocks, learning_rate=0.0001, max_epochs=budget
        ),
        split="trial_80_20" if experiment == "milling" else "shared_80_20",
    )
    if experiment != "milling":
        summary["settings"]["offset"] = offset
    arrays = dict(
        losses=np.asarray(losses),
        test_losses=np.asarray(test_losses),
        parameters=np.full(len(losses), len(weights)),
        weights=weights,
    )
    state = dict(
        gnn=model,
        mean=mean,
        std=std,
        input_size=x.shape[1],
        num_blocks=num_blocks,
        x_train=x_train.numpy(),
        x_test=x_test.numpy(),
        y_train=y_train.numpy().ravel(),
        y_test=y_test.numpy().ravel(),
        train_indices=train_indices,
        test_indices=test_indices,
    )
    if experiment == "milling":
        from milling_data import evaluation

        with torch.no_grad():
            predictions = model(normalized_test).numpy().ravel()
        summary.update(evaluation(predictions, prepared))
        arrays["train_losses_after_update"] = np.asarray(after_losses)
        state["milling_data"] = prepared
    return summary, arrays, state
