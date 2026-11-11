from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

DEFAULT_DATA = Path(__file__).resolve().parent / "data/milling_features.npz"
SENSORS = ("smcAC", "smcDC", "vib_table", "vib_spindle")
WINDOWS = 4


def extract_features(path):

    from scipy.io import loadmat

    trials = np.atleast_1d(loadmat(path, squeeze_me=True, struct_as_record=False)["mill"])
    measured = sorted(
        (trial for trial in trials if np.isfinite(float(trial.VB))),
        key=lambda trial: (int(trial.case), int(trial.run)),
    )
    features = []
    for trial in measured:
        rows = [[] for _ in range(WINDOWS)]
        for sensor in SENSORS:
            signal = np.asarray(getattr(trial, sensor), dtype=float).ravel()
            for row, values in zip(rows, np.array_split(signal, WINDOWS)):
                std = values.std()
                kurtosis = float(np.mean((values-values.mean())**4)/std**4-3) if std else 0.0
                row.extend((std, kurtosis))
        features.append(rows)
    return dict(
        features=np.asarray(features),
        wear=np.asarray([float(trial.VB) for trial in measured]),
        case=np.asarray([int(trial.case) for trial in measured]),
        run=np.asarray([int(trial.run) for trial in measured]),
        raw_trials=np.asarray(len(trials)),
    )


def load_milling(path=None, seed=1234):

    with np.load(path or DEFAULT_DATA, allow_pickle=False) as stored:
        features, wear, cases, runs = (stored[key].copy() for key in ("features", "wear", "case", "run"))
    if features.ndim != 3 or features.shape[1:] != (WINDOWS, 8):
        raise ValueError("Milling features must have shape (trials, 4, 8)")
    order = np.lexsort((runs, cases))
    features, wear, cases, runs = (values[order] for values in (features, wear, cases, runs))
    first = np.flatnonzero(np.r_[True, cases[1:] != cases[:-1]])
    targets = np.setdiff1d(np.arange(len(wear)), first)
    fit, holdout = train_test_split(np.arange(len(targets)), test_size=0.2, random_state=seed)
    fit_rows = set(targets[fit].tolist())
    available = {int(cases[i]): (int(runs[i]), float(wear[i])) for i in first}
    lag, lag_run = [], []
    for i in targets:
        previous_run, previous_wear = available[int(cases[i])]
        lag.append(previous_wear)
        lag_run.append(previous_run)
        if i in fit_rows:
            available[int(cases[i])] = (int(runs[i]), float(wear[i]))
    x = np.concatenate((features[targets], np.broadcast_to(
        np.asarray(lag)[:, None, None], (len(targets), WINDOWS, 1))), axis=2)
    train_sensors = x[fit, :, :8].reshape(-1, 8)
    minimum = train_sensors.min(axis=0)
    span = np.ptp(train_sensors, axis=0)
    span[span == 0] = 1
    x[:, :, :8] = (x[:, :, :8]-minimum)/span
    ids = np.column_stack((cases[targets], runs[targets]))
    return dict(
        x_train=x[fit].reshape(-1, 9), y_train=np.repeat(wear[targets[fit]], WINDOWS),
        x_test=x[holdout].reshape(-1, 9), y_test=np.repeat(wear[targets[holdout]], WINDOWS),
        train_trial_ids=ids[fit], test_trial_ids=ids[holdout],
        train_lag_runs=np.asarray(lag_run)[fit], test_lag_runs=np.asarray(lag_run)[holdout],
        initial_trial_ids=np.column_stack((cases[first], runs[first])),
        feature_minimum=minimum, feature_span=span, windows_per_trial=WINDOWS,
    )


def evaluation(predictions, data):

    values = np.asarray(predictions).ravel()
    trial_prediction = values.reshape(-1, WINDOWS).mean(axis=1)
    trial_target = data["y_test"].reshape(-1, WINDOWS)[:, 0]
    return dict(
        test_window_mse=float(np.mean((values-data["y_test"])**2)),
        test_trial_mse=float(np.mean((trial_prediction-trial_target)**2)),
        train_trials=len(data["train_trial_ids"]), test_trials=len(data["test_trial_ids"]),
        train_windows=len(data["y_train"]), test_windows=len(data["y_test"]),
        windows_per_trial=WINDOWS,
    )
