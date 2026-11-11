# Frequency-radial CWNN

Reusable Algorithms 1 and 2 with optimized frequency radial basis functions, following the radial extension in the Fourier domain described in Section II-C of the associated paper.

```console
python -m pip install -e .
python algorithm1.py data.npz --output results/probe
python train.py data.npz --output results/fit
```

Use Python 3.10 or later. The NPZ file contains `x_train` and `y_train`; `x_test` and `y_test` are optional.

The default learning rate is `0.005`. Set `--learning-rate` and `--mu` as needed. Use `--help` for all options.

See [CITATION.cff](CITATION.cff) and [LICENSING.txt](LICENSING.txt).
