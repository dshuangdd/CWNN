from runtime import Accelerator


def run(settings: dict | None = None, accelerated: bool = True, max_epochs: int = 10000) -> dict:

    settings = {} if settings is None else dict(settings)
    accelerator = Accelerator(enabled=accelerated)
    import torch
    import torch.nn as nn
    import numpy as np
    import itertools
    import time


    @accelerator.torch_class
    class WaveletNet(nn.Module):
        def __init__(self, config, device):
            super().__init__()
            self.parent_map = {}
            self.parent_stage_map = {}
            self.config = config
            self.device = device
            self.j = config["j_init"]
            self.mu = config["mu"]
            self.tran_r = config["tran_r"]
            self.dimension = config["dimension"]
            self.weights = nn.ParameterList()
            self.init_weights = []
            self.centers_list = []
            self.active_indices = []
            self.active_stages = []
            self._init_base_layer()

        def _generate_centers(self, j):

            grid_size = self.tran_r * 2**j + 1
            grid = torch.arange(grid_size, dtype=torch.float32, device=self.device)
            mesh = torch.meshgrid([grid] * self.dimension, indexing="ij")
            return torch.stack(mesh, dim=-1).reshape(-1, self.dimension)

        def energy_separate(self, layer_idx, stage):

            if layer_idx == 0:
                half = len(self.weights[layer_idx]) // 2
                weights = self.weights[layer_idx][half:]
            else:
                weights = self.init_weights[layer_idx]
            energy = weights**2
            (sorted_energy, sorted_indices) = torch.sort(energy, descending=True)
            cum_energy = torch.cumsum(sorted_energy, dim=0)
            total_energy = cum_energy[-1]
            if total_energy == 0:
                return torch.tensor([], device=self.device)
            normalized_cum = cum_energy / total_energy
            max_stage = int(1 / self.mu)
            start_ratio = stage * self.mu
            end_ratio = (stage + 1) * self.mu
            start_idx = torch.searchsorted(normalized_cum, start_ratio)
            end_idx = torch.searchsorted(normalized_cum, end_ratio)
            if stage == max_stage - 1:
                end_idx = len(sorted_indices)
            selected = sorted_indices[start_idx:end_idx]
            return selected

        def _init_base_layer(self):
            centers = self._generate_centers(self.j)
            self.centers_list.append(centers)
            self.active_indices.append(torch.arange(len(centers), device=self.device))
            self.active_stages.append(int(1 / self.mu) - 2)
            self.weights.append(nn.Parameter(torch.zeros(2 * len(centers), device=self.device)))
            self.init_weights.append(torch.zeros(2 * len(centers)))

        def nearest_child_coordinates(self, parent_centers, new_j):
            new_centers = self._generate_centers(new_j)
            selected = []
            grid_step = 1.0
            max_pos = self.tran_r * 2**new_j
            for center in parent_centers:
                dim_coords = []
                for dim in range(self.dimension):
                    coord = (center[dim] / grid_step).round().item()
                    dim_coords.append(coord)
                valid_candidates = []
                for dim in range(self.dimension):
                    base = int(dim_coords[dim])
                    candidates = [base - 1, base, base + 1][:2]
                    valid = list({max(0, min(c, max_pos)) for c in candidates})
                    valid_candidates.append(valid)
                for combo in itertools.product(*valid_candidates):
                    target = (
                        torch.tensor(combo, dtype=torch.float32, device=self.device) * grid_step
                    )
                    mask = torch.all(torch.abs(new_centers - target) < 1e-05, dim=1)
                    indices = torch.nonzero(mask).view(-1)
                    if indices.numel() > 0:
                        selected.append(indices)
            return (
                torch.unique(torch.cat(selected))
                if selected
                else torch.tensor([], device=self.device)
            )

        def wavelet_activation(self, x, j, centers):

            x = x.unsqueeze(1)
            dist = torch.norm(2**j * x - centers, p=2, dim=2)
            y = dist + 1e-06
            return torch.stack(
                [2**j * torch.sin(y) / y, 2**j * (-torch.sin(2 * y) / y + torch.sin(y) / y)]
            )

        def forward(self, x):

            if x.device != self.device:
                x = x.to(self.device)
            output = 0
            for layer_idx in range(len(self.weights)):
                active_idx = self.active_indices[layer_idx]
                assert active_idx.max() < len(
                    self.centers_list[layer_idx]
                ), f"Center index outside layer {layer_idx}"
                assert active_idx.min() >= 0, f"Negative center index in layer {layer_idx}"
                centers = self.centers_list[layer_idx][self.active_indices[layer_idx]]
                weights = self.weights[layer_idx]
                current_j = self.j + layer_idx
                x_exp = x.unsqueeze(1)
                dist = torch.norm(2**current_j * x_exp - centers, p=2, dim=2)
                y = dist + 1e-06
                if layer_idx == 0:
                    phi_activ = 2**current_j * torch.sin(y) / y
                    psi_activ = 2**current_j * (-torch.sin(2 * y) / y + torch.sin(y) / y)
                    activ = torch.stack([phi_activ, psi_activ])
                    output += (
                        activ[0] @ weights[: len(active_idx)]
                        + activ[1] @ weights[len(active_idx) :]
                    )
                else:
                    psi_activ = 2**current_j * (-torch.sin(2 * y) / y + torch.sin(y) / y)
                    output += psi_activ @ weights[: len(active_idx)]
            return output

        def add_layer(self):
            current_layer = len(self.weights) - 1
            if self.active_stages[current_layer] < int(1 / self.mu) - 1:
                self.active_stages[current_layer] += 1
                current_stage = self.active_stages[current_layer]
                if current_layer in self.parent_stage_map:
                    parent_layer = self.parent_stage_map[current_layer]["parent"]
                    parent_stage = self.parent_stage_map[current_layer]["stages"][current_stage]
                    parent_idx = self.energy_separate(parent_layer, parent_stage)
                    new_j = self.j + current_layer
                    selected_idx = self.nearest_child_coordinates(
                        self.centers_list[parent_layer][parent_idx], new_j
                    )
                    combined_idx = torch.cat([self.active_indices[current_layer], selected_idx])
                    self.active_indices[current_layer] = torch.unique(combined_idx)
                    if (
                        self.active_stages[current_layer] == int(1 / self.mu) - 1
                        and current_layer > 0
                    ):
                        total = len(self.centers_list[current_layer])
                        self.active_indices[current_layer] = torch.arange(total, device=self.device)
            else:
                parent_layer = current_layer
                new_layer_idx = len(self.weights)
                new_j = self.j + new_layer_idx
                new_centers = self._generate_centers(new_j)
                max_stage = int(1 / self.mu)
                self.parent_stage_map[new_layer_idx] = {
                    "parent": parent_layer,
                    "stages": {i: i for i in range(max_stage)},
                }
                self.init_weights[parent_layer] = self.weights[parent_layer].clone()
                parent_stage = self.parent_stage_map[new_layer_idx]["stages"][0]
                if current_layer >= 0 and self.mu == 1:
                    total = len(self.centers_list[current_layer])
                    selected_idx = torch.arange(total, device=self.device)
                else:
                    parent_idx = self.energy_separate(parent_layer, parent_stage)
                    selected_idx = self.nearest_child_coordinates(
                        self.centers_list[parent_layer][parent_idx], new_j
                    )
                self.centers_list.append(new_centers)
                self.active_indices.append(selected_idx)
                self.active_stages.append(0)
                param_size = 2 * len(new_centers) if len(self.weights) == 0 else len(new_centers)
                self.weights.append(nn.Parameter(torch.zeros(param_size, device=self.device)))
                self.init_weights.append(torch.zeros(param_size, device=self.device))

        def get_activation_status(self):

            report = []
            for layer_idx in range(len(self.weights)):
                if layer_idx == 0:
                    total = len(self.centers_list[layer_idx])
                else:
                    total = len(self.centers_list[layer_idx])
                activated = len(self.active_indices[layer_idx])
                stages = self.active_stages[layer_idx] + 1 if layer_idx > 0 else 1
                report.append(
                    f"Layer {layer_idx}: {activated}/{total} (stage {stages}/{int(1 / self.mu)})"
                )
            return "\n".join(report)

    class DataProcessor:
        def __init__(self, config, device):
            self.config = config
            self.device = device

        def load(self):
            from milling_data import load_milling

            self.data = load_milling(self.config.get("data_path"))
            return tuple(torch.tensor(self.data[key], dtype=torch.float32, device=self.device)
                         for key in ("x_train", "y_train", "x_test", "y_test"))


    def train_network(model, X_train, y_train, X_test, y_test, config):

        optimizer = torch.optim.AdamW([{"params": model.weights[0]}], lr=config["learn_rate"])
        loss_fn = nn.MSELoss().to(model.device)
        history = {
            "train_loss": [],
            "train_loss_after_update": [],
            "test_loss": [],
            "delta_loss": [],
            "active_units": [],
            "params_count": [],
        }
        previous_train_loss = float("inf")
        start_time = time.time()
        for epoch in range(config["epochs"]):
            model.train()
            optimizer.zero_grad()
            pred = model(X_train)
            loss = loss_fn(pred, y_train)
            loss.backward()
            for (name, param) in model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        raise ValueError(f"Gradient for {name} contains NaN")
                    if torch.isinf(param.grad).any():
                        raise ValueError(f"Gradient for {name} contains Inf")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.eval()
            with torch.no_grad():
                current_train_loss = loss_fn(model(X_train), y_train)
                test_pred = model(X_test)
                test_loss = loss_fn(test_pred, y_test)
            delta_loss = abs(loss.item() - previous_train_loss)
            previous_train_loss = loss.item()
            current_params = sum((p.numel() for p in model.parameters()))
            current_units = [len(idx) for idx in model.active_indices]
            history["train_loss"].append(loss.item())
            history["train_loss_after_update"].append(current_train_loss.item())
            history["test_loss"].append(test_loss.item())
            history["delta_loss"].append(delta_loss)
            history["active_units"].append(current_units)
            history["params_count"].append(current_params)
            if current_train_loss < config["pre_ac"]:
                break
            if delta_loss < config["pre_term"] and len(model.weights) < config["max_layers"]:
                prev_total_params = sum((p.numel() for p in model.parameters()))
                model.add_layer()
                new_param = model.weights[-1]
                param_exists = any(
                    (
                        torch.equal(p, new_param)
                        for group in optimizer.param_groups
                        for p in group["params"]
                    )
                )
                if not param_exists:
                    optimizer.add_param_group({"params": new_param})
                new_layer = model.weights[-1]
                new_param_count = new_layer.numel()
                active_params = 0
                for (i, active_idx) in enumerate(model.active_indices):
                    if i == 0:
                        active_params += 2 * len(active_idx)
                    else:
                        active_params += len(active_idx)
                current_active = sum((len(idx) for idx in model.active_indices))
                prev_active = history["active_units"][-1][-1] if history["active_units"] else 0
        end_time = time.time()
        time_long = end_time - start_time
        epoch_train = len(history["train_loss"])
        return history

    config = dict(
        j_init=0,
        mu=1 / 3,
        tran_r=1,
        dimension=9,
        learn_rate=0.001,
        test_size=0.2,
        random_state=1234,
        pre_ac=0.015,
        pre_term=4e-05,
        max_layers=3,
        epochs=max_epochs,
    )
    config.update(settings)
    device = torch.device("cpu")
    torch.set_num_threads(1)
    torch.manual_seed(1234)
    processor = DataProcessor(config, device)
    (X_train, y_train, X_test, y_test) = processor.load()
    model = WaveletNet(config, device)
    history = train_network(model, X_train, y_train, X_test, y_test, config)
    return locals()
