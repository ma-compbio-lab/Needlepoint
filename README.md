# Needlepoint

Needlepoint models spatial molecular expression with global V-SKI and local
VNNGP Gaussian processes, groups related fields into uncertainty-aware spatial
prototypes, and infers spatial domains with a latent block model (LBM).

Library version: **0.2.0**.

## Installation

Use Python 3.11 or 3.12. Clone the repository:

```bash
git clone https://github.com/yangyuanz/needlepoint-software.git
cd needlepoint-software
```

Then install:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install '.[gp]'
needlepoint --help
```

The base profile (`python -m pip install .`) provides input readers,
normalization, and CPU scoring. The `gp` extra adds models, prototypes, and LBM
dependencies. Optional `parquet` and `zarr` extras support their input formats.
For CUDA, install the appropriate Torch build before installing `.[gp]`.

## Documentation

- [User guide](docs/USER_GUIDE.md): installation, inputs, complete workflows,
  configuration, outputs, resource planning, and troubleshooting.
- [API reference](docs/API.md): all top-level Python exports, signatures,
  settings, return values, module map, and command-line entry points.
- [Implementation](src/needlepoint): Python source and its reference
  policy resource.

## Source

The implementation is based on Needlepoint commit
[`e695b592ea0271d286d8da4af357c4d7255c9d64`](https://github.com/yangyuanz/needlepoint/tree/e695b592ea0271d286d8da4af357c4d7255c9d64).
Saved runs record their source identity and must use the matching source
snapshot when resumed.

## Licensing

A project license has not yet been selected. The GPyTorch notice is in
[LICENSES/GPyTorch-MIT.txt](LICENSES/GPyTorch-MIT.txt).
