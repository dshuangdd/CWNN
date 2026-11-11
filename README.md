# CWNN

Code for [Optimizing Basis Function Selection in Constructive Wavelet Neural Networks and Its Applications](https://doi.org/10.1109/TAI.2025.3621177).

**V1** focuses on reproducing the paper's four experiments. **V2** provides general implementations of Algorithms 1 and 2, optimized for computational efficiency and reuse.

| Directory | Contents |
|---|---|
| `CWNN_V1_Reproduction` | Paper experiments and comparison models. |
| `CWNN_V2_SpatialRadial` | Optimized V2 with spatial radial sinc basis functions. |
| `CWNN_V2_FrequencyRadial` | Optimized V2 with frequency radial basis functions. |

V2 offers two basis constructions for further development and applications. The frequency radial version follows the Fourier-domain radial extension described in Section II-C of the paper. The spatial radial version uses a direct spatial radial sinc construction for computational simplicity. Both use optimized numerical routines and Adam training. Readers can choose either version according to their application and computational requirements; numerical results can differ.

## Run

Use Python 3.10 or later.

V1: run from the repository root.

```console
cd CWNN_V1_Reproduction
python -m pip install -r requirements.txt
python example1.py
```

Use `example2.py`, `example3.py`, or `example4.py` for the other experiments. Algorithm 1, comparison models and additional experiment scripts are included.

V2: open either `CWNN_V2_SpatialRadial` or `CWNN_V2_FrequencyRadial` and run the following commands. Use one V2 version per Python environment.

```console
python -m pip install -e .
python algorithm1.py data.npz --output results/probe
python train.py data.npz --output results/fit
```

V2 accepts NPZ data containing `x_train` and `y_train`, with optional `x_test` and `y_test`.

The default learning rate is `0.0005` for spatial radial V2 and `0.005` for frequency radial V2. Use `--learning-rate`, `--mu`, and `--help` to set parameters and view available options.

Citation: [CITATION.cff](CITATION.cff). Licensing is described in each version's `LICENSING.txt`; dataset attribution is in [SOURCE.txt](CWNN_V1_Reproduction/data/SOURCE.txt).
