# Needlepoint user guide

This guide describes Needlepoint **0.2.0**.
See the [API reference](API.md) for exact signatures and settings.

## 1. Software utility

Needlepoint links three spatial representations:

| Representation | Purpose | Typical input |
| --- | --- | --- |
| Per-feature expression field | Estimate spatial signal and posterior variance | Molecular observations and physical XY coordinates |
| Spatial prototype | Group features with related spatial field shapes | Fitted feature fields and their variances |
| Spatial domain | Group locations with related molecular block signatures | Fields, prototypes, and a spatial graph |

The default count workflow performs full-library CP10K/log1p preprocessing,
multiscale feature scoring, per-feature length-scale initialization, joint
global/local GP fitting, prototype fitting/merging, and fixed-domain LBM fits.
Each stage can also be called separately. Domain IDs and prototype IDs are
fit-specific; they do not automatically identify a biological cell type.

## 2. Installation

Use Python 3.11 or 3.12 in a virtual environment. Linux is the primary
platform. From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install '.[gp]'
```

For a CPU-only Torch installation, install Torch before Needlepoint:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install '.[gp]'
```

For CUDA, install a Torch build appropriate to the allocated GPU environment
before `.[gp]`. Needlepoint uses visible CUDA ordinals and honors the scheduler's
`CUDA_VISIBLE_DEVICES` allocation. It does not allocate a GPU or install drivers.

| Install command | Capability |
| --- | --- |
| `python -m pip install .` | Readers, normalization, CPU scoring |
| `python -m pip install '.[gp]'` | Also Gaussian/coupled-VI models, prototypes, and LBM dependencies |
| `python -m pip install '.[parquet]'` | Also Parquet input |
| `python -m pip install '.[zarr]'` | Also AnnData Zarr input |
| `python -m pip install '.[gp,zarr]'` | Model dependencies plus both optional file formats; `gp` includes PyArrow |

Install from this repository to use this implementation. A plain
`import needlepoint` keeps Torch and GPyTorch imports lazy.

## 3. Input contract

All matrix readers return `SpatialData` with these aligned fields:

| Field | Contract |
| --- | --- |
| `counts` | Copied CSR sparse matrix, locations × features |
| `coords` | Finite locations × 2 array, XY in micrometres |
| `genes` | Unique, nonempty feature names in matrix-column order |
| `obs`, `var` | Observation and feature annotations in the same order |
| `measurement` | `"counts"` or `"continuous"` |
| `metadata` | Input provenance |

Count observations must be finite and nonnegative. Continuous observations may
be signed. Readers preserve supplied zero-count locations and do not normalize
or select features. Supply the full measured library before feature selection
so normalization uses the correct library totals. Preserve the full acquired
support, including empty bins; omitted acquisition bins cannot be recovered by
a matrix reader.

### H5AD or AnnData

```python
import needlepoint as ndl

data = ndl.read_spatial(
    "section.h5ad",
    layer="counts",          # Use layer=None when X contains raw counts.
    coord_key="spatial",     # An obsm key, or a pair such as "x_um,y_um".
    coordinate_scale=1.0,     # Multiplier converting both axes to micrometres.
)
```

`read_spatial` also accepts AnnData objects, AnnData Zarr paths, dense/sparse
matrices, and SciPy sparse NPZ matrices. Matrix and NPZ inputs require explicit
coordinates and feature names:

```python
data = ndl.read_spatial(count_matrix, coords=xy_um, genes=gene_names)
data = ndl.read_spatial("counts.npz", coords=xy_um, genes=gene_names)
```

Input loading currently uses RAM. The common reader is not a backed,
out-of-core ingestion system. It retains the selected matrix, coordinates,
`obs`, and `var`; it does not copy all images, embeddings, layers, or `uns` fields.

### 10x counts

```python
data = ndl.read_10x_mtx(
    "raw_feature_bc_matrix",
    positions="positions_in_microns.csv",
    barcode="barcode", x="x_um", y="y_um",
    feature_column=0,
)

data = ndl.read_10x_h5(
    "raw_feature_bc_matrix.h5",
    positions="positions_in_microns.csv",
    barcode="barcode", x="x_um", y="y_um",
)
```

Positions align by barcode. Every input barcode, including an empty one, needs
a coordinate. MTX/TSV files may be gzipped. Feature IDs are safer than symbols
when symbols are duplicated. Image scale factors and coordinate registration
must be applied explicitly.

### Transcript tables and GEM

```python
data = ndl.read_transcripts(
    "transcripts.parquet",
    bounds=(0.0, 0.0, 1000.0, 1000.0),
    pitch=10.0,
    genes=full_assay_panel,
    x="x_location", y="y_location", gene="feature_name",
    quality="qv", minimum_quality=20,
)

data = ndl.read_gem(
    "section.gem.gz",
    bounds=(0.0, 0.0, 1000.0, 1000.0),
    pitch=10.0,
    genes=full_assay_panel,
    coordinate_scale=micrometres_per_source_unit,
)
```

Bounds use `(x_min, y_min, x_max, y_max)` in micrometres, with half-open upper
edges. Side lengths must be integer multiples of pitch. These readers create
the complete rectangular grid, including empty bins, and use bin-centre
coordinates. A supplied full gene panel excludes control features by identity.
Grid size grows with area divided by pitch squared; plan memory accordingly.
For event-level LGCP modeling, use `non_gaussian.read_point_pattern` and
`fit_lgcp` to preserve individual events rather than converting them to counts.

## 4. Complete workflow

```python
import needlepoint as ndl

data = ndl.read_spatial("section.h5ad", layer="counts", coord_key="spatial")
workflow = ndl.Workflow(
    data,
    "outputs/section",
    resources=ndl.Resources(device="auto", cpu_threads=4),
    config=ndl.WorkflowConfig(grid_size=128, domain_counts=(4, 8, 12)),
)
plan = workflow.plan()
result = workflow.run()

gp_maps = result.gp_dir
prototypes = result.prototype_dir
domains_r8 = result.domain_dir(8)
```

`plan()` reports the stage commands, source/input/configuration identities,
resource allocation, and storage estimates without creating the output
directory. It loads the input and may query the selected device. `run()`
executes the stages sequentially in subprocesses using a frozen package copy.
Use a new output directory for a new run.

For a fixed panel, set `selected_genes=("GENE_A", "GENE_B", ...)` explicitly.
The panel must be unique and fully present. Selection bypasses discovery
scoring; normalization still uses the full supplied library. Set
`n_prototypes` to a valid count for the panel when running downstream stages.
For discovery, all genes passing the empirical-FDR/artifact criteria are
retained; no fixed panel size is imposed.

Use `stop_after="score"`, `"screen"`, `"gp"`, or `"prototypes"` when later
stages are unnecessary. The default is `"lbm"`.

### Command-line workflow

```bash
needlepoint run --h5ad section.h5ad --layer counts --output-dir outputs/section
needlepoint run --h5ad section.h5ad --layer counts --output-dir outputs/section --execute
needlepoint run --h5ad section.h5ad --layer counts --output-dir outputs/section --execute --resume
```

The first command prints a plan. `--execute` starts processing. `--gene-panel`
accepts a CSV with a `gene` column. `--stop-after` and `--device` override the
corresponding configuration options.

For reusable configuration, save this as your own `workflow.toml`:

```toml
[workflow]
grid_size = 128
stop_after = "lbm"
domain_counts = [4, 8, 12]
n_prototypes = 20
global_epochs = 30
local_epochs = 30
joint_epochs = 10

[resources]
device = "auto"
cpu_threads = 4
memory_fraction = 0.7
```

```bash
needlepoint run --h5ad section.h5ad --layer counts --config workflow.toml --output-dir outputs/section --execute
```

`needlepoint defaults` prints a descriptive reference policy. It is separate
from the executable `[workflow]`/`[resources]` format above. Low-level
constructors do not automatically consume that reference policy.

## 5. Scientific settings

### Preprocessing and observation model

The default Gaussian workflow fits full-library `log1p(CP10k)` expression.
If preprocessing separately, use `ndl.normalize_counts(raw_counts)` before
subsetting genes. The function returns a new CSR matrix and preserves empty
rows. It does not alter the original counts.

The opt-in `WorkflowConfig(observation_model="poisson-log")` fits raw counts
with full-library exposure offsets. Its length-scale screen still uses
Gaussian likelihood on `log1p(CP10k)`. Zero-exposure locations remain in the
reported support. This Poisson workflow is experimental.

Continuous measurements use `read_spatial(..., measurement="continuous")`
and the low-level `fit_gp` interface. The high-level Workflow accepts counts.
The explicit coupled-VI interface additionally supports Tobit censoring,
Poisson counts, and event-preserving LGCP. Tobit requires censor flags,
threshold, noise scale, and a measurement-scale description. LGCP requires
acquired-window bounds, quadrature areas, and effort. These optional methods
retain their experimental status and convergence diagnostics.

### Physical length scales

The default `physical_multiscale_v1` policy uses micrometre coordinates:

- Set the local Matérn scale to `max(10 micrometres, sampling pitch)`.
- Screen global RBF scales with a lower bound of
  `max(global_min_length_scale, 4 * local_scale)`.
- Use half the larger field-of-view span for the initial upper bound, with
  at least two candidates and a default cap of eight.

Supply `sampling_interval_um` for known acquisition pitch or irregular cells.
An estimated nearest-neighbour distance does not establish assay resolution.
The policy does not automatically enlarge the inducing grid.

For historical behavior, choose `length_scale_policy="grid_aware_log2"`.
For standalone screen/fit commands, use the same policy in both stages.

### Main workflow defaults

| Setting | Default |
| --- | --- |
| Normalization | Full-library target sum 10,000, then log1p |
| Scoring bandwidths | 20, 40, 80, 160, 320, 640 micrometres |
| Empirical FDR / maximum artifact-like score | 0.05 / 0.35 |
| Inducing grid / local neighbours | 128 per axis / 16 |
| Global, local, joint training epochs | 30, 30, 10 |
| Initial prototypes / starts / iteration cap | 20 / 3 / 20 |
| Prototype merge map-correlation threshold | 0.90 |
| Domain counts / LBM iteration cap | 4, 8, 12 / 30 |
| Length-scale policy / observation model | `physical_multiscale_v1` / `gaussian` |

Epoch and iteration limits are budgets. Inspect the resulting histories and
convergence metadata before drawing scientific conclusions. The functional
prototype optimizer uses the versioned normalized, precision-weighted centroid
correction retained from upstream. Increasing group count can improve fit
without establishing biological specificity.

## 6. Independent stages

Use installed help for each stage's full argument list:

```bash
needlepoint score --help
needlepoint fit --help
needlepoint prototypes --help
needlepoint lbm --help
needlepoint fit-non-gaussian --help
needlepoint-condition-gaussian --help
```

Normal screening and fitting share `needlepoint fit --mode screen` and
`needlepoint fit --mode fit`. In contrast, `needlepoint screen` selects the
explicit optional deterministic direct-SKI screen. It is not the Workflow's
default screening backend.

The standalone fit adapter defaults to evaluation with 2% random and 2%
blocked holdouts. For downstream production maps, explicitly set
`--fit-purpose production --random-holdout-fraction 0 --blocked-holdout-fraction 0`.
Workflow sets these production options itself. Production coverage validation
is required by the prototype/LBM adapters; evaluation outputs are separate.

Dedicated commands also include `needlepoint-merge-gp-shards`,
`needlepoint-merge-lengthscale-screens`, and `needlepoint-provenance`.
All input/output paths are caller supplied. See the [command table](API.md#command-line-interface).

## 7. Outputs and interpretation

| Location under the workflow output directory | Content |
| --- | --- |
| `input.h5ad` | Canonical input copy, or explicitly requested reference |
| `workflow_plan.json`, `workflow_state.json` | Run identity, stage state, and completed-file hashes |
| `source/needlepoint/`, `logs/` | Frozen runtime source and per-stage logs |
| `scoring/` | Scores and selected panel; omitted in fixed-panel mode |
| `screen/` | Initial length-scale table and completion summary |
| `gp/fitted_maps/` | Chunked posterior maps, feature metadata, coordinates, manifest |
| `prototypes/` | Prototype maps, memberships, assignments, history, and manifest |
| `lbm/R4/`, `lbm/R8/`, `lbm/R12/` | Domain fits for the requested counts |

GP chunks use the canonical `posterior_mean`/`posterior_variance` fields with
explicit feature and spot order. Default Gaussian exports retain an FP16
moment pair with overflow validation. Computation remains in its configured
precision. Component, predictive, latent, and cache retention are opt-in.
Coupled-VI exports default to FP32 and require explicit field scale/units.
Latent variance, uncertainty in a rate, and observation variance including
sampling noise are different quantities.

For Gaussian fits, posterior variance does not itself establish calibrated
biological uncertainty. For coupled conditioning, global/local cross-covariance
must be handled consistently. Preserve manifests, feature metadata, and spot
tables with the numerical arrays; arrays alone do not establish support or
downstream eligibility.

## 8. Resources, storage, and resume

`Resources(device="auto")` chooses a visible CUDA device or CPU.
`device="cpu"` selects CPU explicitly; an unavailable explicit CUDA request
fails. The executor uses one visible GPU or CPU. Slurm submission and
automatic multi-GPU queues are outside this interface.

Tune `feature_chunk_size`, `batch_size`, `local_kl_batch_size`, and
`cpu_threads` when needed. Resource choices do not truncate the feature panel
or acquired support. Estimates exclude the resident sparse input and are
planning heuristics, not capacity guarantees. Two FP16 posterior arrays alone
need approximately `4 * n_locations * n_features` bytes; allow additional
space for metadata, temporary files, source snapshots, and downstream outputs.

`input_storage="reference"` avoids another disk copy only for an existing
canonical H5AD with raw counts in `X` and micrometre XY coordinates in
`obsm["spatial"]`. It still loads the input into memory and requires the
referenced file to remain unchanged. `retain_caches=True` preserves optional
working caches that would otherwise be removed after successful GP fitting.

```python
result = workflow.run(resume=True)
```

Resume requires matching inputs, configuration, dependencies, resources, and
source identity. Completed files are verified by hash. Incomplete stage
directories are preserved with timestamped names before retry. If the plan
or source has changed, use a new output directory. A saved run must continue
using the source snapshot that created it.

## 9. Troubleshooting and limits

| Symptom | Action |
| --- | --- |
| Missing Torch/GPyTorch or Parquet reader | Install the corresponding `gp` or `parquet` extra into the active environment |
| Duplicate genes, missing barcodes, misaligned coordinates | Align the matrix and annotations explicitly; retain unique identifiers |
| Too few discovery genes | Inspect full-library input, physical units, scores, and selection thresholds |
| Prototype count exceeds a fixed panel | Choose a valid `n_prototypes` explicitly |
| CUDA unavailable or ordinal outside the allocation | Use CPU or an ordinal visible within the scheduler allocation |
| Memory pressure | Lower feature/spot batches and plan for the resident input matrix |
| Existing nonempty output directory | Resume a matching run or choose a new directory |
| GP artifacts rejected downstream | Use full-support production fitting and preserve the complete metadata package |
| Iteration cap or unconverged result | Inspect training history and diagnostics; completion alone does not prove convergence |

Optional covariance, non-Gaussian, factor, direct-SKI, and alternative solver
implementations remain opt-in. Their presence does not change the default
workflow or establish a validation claim for a new dataset. The distribution
contains no laboratory data or fitted results. Licensing is described in the
[README](../README.md#licensing).
