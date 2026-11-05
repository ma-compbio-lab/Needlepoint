# Needlepoint API reference

This reference covers every top-level export in `needlepoint.__all__` for
Needlepoint **0.2.0**. Signatures and setting defaults
come from the included source. See the [user guide](USER_GUIDE.md) for
installation, full workflows, inputs, and output interpretation.

```python
import needlepoint as ndl
```

Model exports are lazy and require the `gp` extra. The constant
`ndl.__version__` is `"0.2.0"`. Low-level settings and Workflow production
settings are separate; reference-policy TOML is descriptive.
Signatures below use reference notation; `<factory>` denotes a fresh per-instance
default, such as the empty metadata dictionary on SpatialData.

## Inputs and preprocessing

### `SpatialData`

Aligned CSR measurements, physical XY coordinates, feature identities, and observation/feature annotations. Also exposes shape, to_anndata(), write_h5ad(path), and fingerprint().

Measurements by locations, retaining all supplied observation rows.

Coordinates are physical XY micrometres. ``measurement`` records whether
values are counts or continuous measurements; conversion never normalizes.

[Implementation](../src/needlepoint/inputs.py).

```python
ndl.SpatialData(
    counts: 'object',
    coords: 'np.ndarray',
    genes: 'object',
    obs: 'pd.DataFrame | None' = None,
    var: 'pd.DataFrame | None' = None,
    measurement: 'str' = 'counts',
    metadata: 'dict' = <factory>,
) -> None
```

```python
SpatialData.to_anndata(
    self,
)
```

```python
SpatialData.write_h5ad(
    self,
    path,
)
```

```python
SpatialData.fingerprint(
    self,
)
```

### `read_spatial`

Return SpatialData. Matrix/NPZ inputs require coords and genes. Input loading uses RAM and performs no normalization.

Read AnnData/H5AD or a dense/sparse matrix with explicit XY coordinates.

``coordinate_scale`` converts supplied coordinates to micrometres. No
coordinate units are inferred, and no observation or feature is removed.
``.npz`` paths are SciPy sparse-matrix files and require coords and genes.
Saved measurement semantics are preserved. Conflicting explicit semantics
require ``allow_measurement_override=True``. Unlabelled matrices default
to counts; ingestion never infers the likelihood from a platform name.

[Implementation](../src/needlepoint/inputs.py).

```python
ndl.read_spatial(
    source,
    *,
    coords=None,
    genes=None,
    layer=None,
    coord_key='auto',
    coordinate_scale=1.0,
    measurement=None,
    obs=None,
    allow_measurement_override=False,
)
```

### `read_10x_mtx`

Return SpatialData, aligned by barcode and preserving empty observations. Plain/gzipped MTX/TSV files are supported.

Read 10x features × barcodes MTX and align positions by barcode.

Positions must be a DataFrame or CSV with explicitly named columns. Supply
feature-column 1 for gene symbols only when unique; IDs (column 0) are the
default. Every barcode requires coordinates, including zero-count columns.

[Implementation](../src/needlepoint/inputs.py).

```python
ndl.read_10x_mtx(
    directory,
    *,
    positions,
    x='x',
    y='y',
    barcode='barcode',
    coordinate_scale=1.0,
    feature_column=0,
)
```

### `read_10x_h5`

Return SpatialData from modern 10x feature-barcode HDF5; coordinates align by barcode.

Read the modern 10x /matrix CSC HDF5 layout; align all barcodes by ID.

Supply physical positions (DataFrame/CSV/Parquet) and any unit conversion.
This is a feature-barcode reader, not a general vendor-HDF5 reader.

[Implementation](../src/needlepoint/inputs.py).

```python
ndl.read_10x_h5(
    path,
    *,
    positions,
    x='x',
    y='y',
    barcode='barcode',
    coordinate_scale=1.0,
    feature_key='id',
)
```

### `read_gem`

Return full-grid SpatialData. Physical acquisition bounds, pitch, full gene panel, and coordinate conversion are explicit.

Bin a Stereo-seq GEM count table on explicit full physical support.

GEM coordinate units vary by acquisition; coordinate_scale is mandatory.
Bounds/pitch are XYXY micrometres. Weighted entries are summed directly,
without expanding counts into repeated molecules. Empty bins are retained.

[Implementation](../src/needlepoint/inputs.py).

```python
ndl.read_gem(
    source,
    *,
    bounds,
    pitch,
    genes,
    coordinate_scale,
    gene='geneID',
    count=None,
    x='x',
    y='y',
)
```

### `read_transcripts`

Return full-grid SpatialData. Bounds use (x_min, y_min, x_max, y_max), with half-open upper edges. LGCP uses the separate event reader.

Bin a transcript DataFrame/CSV/Parquet on an explicit full XY rectangle.

Bounds and pitch are in micrometres. The supplied complete gene panel
defines controls to exclude. Quality filtering and exclusion counts are
recorded; every grid bin, including empty bins, is retained.

[Implementation](../src/needlepoint/inputs.py).

```python
ndl.read_transcripts(
    source,
    *,
    bounds,
    pitch,
    genes,
    x='x_location',
    y='y_location',
    gene='feature_name',
    quality='qv',
    minimum_quality=20,
    coordinate_scale=1.0,
)
```

### `normalize_counts`

Return a new CSR matrix after full-library target-sum normalization and log1p. Retain empty rows and leave the input unchanged. Normalize before selecting features.

Copy counts to CSR, apply full-library CP10K/log1p, and retain zero rows.

Supply the complete measured gene matrix, *before* selecting a gene panel.
Coordinates and observation ordering are never changed. Negative or nonfinite
values are rejected. Input is not modified.

[Implementation](../src/needlepoint/api.py).

```python
ndl.normalize_counts(
    counts,
    *,
    target_sum=10000.0,
)
```

### `SpatialFeatureDataset`

Locations x features matrix plus spatial metadata.

The proposal writes the matrix as features x locations. The executable code
stores `X` as locations x features because that is the AnnData/scikit-learn
convention. `feature_names` therefore index columns and `obs` indexes rows.

[Implementation](../src/needlepoint/data.py).

```python
ndl.SpatialFeatureDataset(
    X: 'sparse.spmatrix | np.ndarray',
    coords: 'np.ndarray',
    feature_names: 'np.ndarray',
    obs: 'pd.DataFrame | None' = None,
    feature_metadata: 'pd.DataFrame | None' = None,
    ground_truth: 'np.ndarray | None' = None,
    name: 'str' = 'dataset',
) -> None
```

## Workflow and resources

### `WorkflowConfig`

Executable count-workflow settings. selected_genes chooses an exact panel; stop_after is score, screen, gp, prototypes, or lbm. The table gives implementation defaults.

Sequencing-count workflow policy; physical scales are micrometres.

Explicit selected_genes bypasses discovery scoring for matched-panel
experiments. Resource choices never truncate an FDR-selected panel.

[Implementation](../src/needlepoint/workflow.py).

| Parameter | Type | Default |
| --- | --- | --- |
| `selected_genes` | `tuple[str, ...] \| None` | `None` |
| `stop_after` | `str` | `'lbm'` |
| `grid_size` | `int` | `128` |
| `bandwidths_um` | `tuple[float, ...]` | `(20.0, 40.0, 80.0, 160.0, 320.0, 640.0)` |
| `target_sum` | `float` | `10000.0` |
| `fdr_q` | `float` | `0.05` |
| `max_artifact_like_score` | `float` | `0.35` |
| `min_spots` | `int` | `20` |
| `minimum_selected_genes` | `int` | `20` |
| `null_profiles_per_stratum` | `int` | `2000` |
| `global_epochs` | `int` | `30` |
| `local_epochs` | `int` | `30` |
| `joint_epochs` | `int` | `10` |
| `global_min_length_scale` | `float` | `40.0` |
| `global_initial_length_scale` | `float` | `80.0` |
| `local_min_length_scale` | `float` | `8.0` |
| `local_initial_length_scale` | `float` | `16.0` |
| `local_max_length_scale` | `float` | `32.0` |
| `local_neighbors` | `int` | `16` |
| `n_prototypes` | `int` | `20` |
| `prototype_iterations` | `int` | `20` |
| `prototype_n_init` | `int` | `3` |
| `prototype_merge_correlation` | `float` | `0.9` |
| `domain_counts` | `tuple[int, ...]` | `(4, 8, 12)` |
| `lbm_iterations` | `int` | `30` |
| `random_state` | `int` | `0` |
| `retain_caches` | `bool` | `False` |
| `input_storage` | `str` | `'copy'` |
| `length_scale_policy` | `str` | `'physical_multiscale_v1'` |
| `sampling_interval_um` | `float \| None` | `None` |
| `physical_local_floor_um` | `float` | `10.0` |
| `physical_global_to_local_ratio` | `float` | `4.0` |
| `physical_upper_fov_fraction` | `float` | `0.5` |
| `physical_max_candidates` | `int` | `8` |
| `observation_model` | `str` | `'gaussian'` |

### `Resources`

CPU or one visible CUDA device. resolve() returns conservative starting allocations and memory estimates; estimates exclude resident input and are not capacity guarantees.

[Implementation](../src/needlepoint/resources.py).

| Parameter | Type | Default |
| --- | --- | --- |
| `device` | `str` | `'auto'` |
| `cpu_threads` | `int` | `1` |
| `memory_fraction` | `float` | `0.7` |
| `feature_chunk_size` | `int \| None` | `None` |
| `batch_size` | `int \| None` | `None` |
| `local_kl_batch_size` | `int` | `1024` |

```python
Resources.resolve(
    self,
    *,
    n_spots,
    n_features,
    grid_size,
)
```

### `Workflow`

plan() reports commands and identities without writing run outputs. run() executes a new run; run(resume=True) requires matching identity and verified completed files.

Plan or run discovery → screen → full-support GP → prototypes → LBM.

Each stage runs in a separate process using the installed package. This
avoids global argv changes and lets callers inspect commands and logs.
This initial executor supports one visible GPU or CPU; scheduler allocation
stays under caller control. Ingestion currently loads the sparse matrix in
RAM. Lower-level module APIs remain available for other measurement types.

[Implementation](../src/needlepoint/workflow.py).

```python
ndl.Workflow(
    data,
    output_dir,
    *,
    config=None,
    resources=None,
    **read_options,
)
```

```python
Workflow.plan(
    self,
)
```

```python
Workflow.run(
    self,
    *,
    resume=False,
)
```

### `WorkflowResult`

Run descriptor. gp_dir, prototype_dir, and domain_dir(n_domains) point to the corresponding stages; only requested stages exist.

[Implementation](../src/needlepoint/workflow.py).

```python
ndl.WorkflowResult(
    output_dir: 'Path',
    state: 'dict',
) -> None
```

```python
WorkflowResult.domain_dir(
    self,
    n_domains,
)
```

### `default_workflow`

Return a fresh dictionary containing the descriptive reference policy. It is separate from executable WorkflowConfig and is not injected into model constructors. Campaign-only profiles are omitted here.

Return a fresh copy of the reference policy (not automatically applied to fits).

[Implementation](../src/needlepoint/api.py).

```python
ndl.default_workflow()
```

## Gaussian spatial fields

### `JointGlobalLocalGPSettings`

Low-level model settings are separate from WorkflowConfig. Arrange preprocessing, feature selection, and length-scale screening explicitly for low-level fitting.

Settings for additive global/local GP joint variational training.

[Implementation](../src/needlepoint/gpytorch_joint_global_local.py).

| Parameter | Type | Default |
| --- | --- | --- |
| `output_dtype` | `str` | `'float16'` |
| `retain_predictive_variance` | `bool` | `False` |
| `retain_component_maps` | `bool` | `False` |
| `retain_latent_maps` | `bool` | `False` |
| `global_grid_size` | `int` | `128` |
| `global_epochs` | `int` | `5` |
| `local_epochs` | `int` | `5` |
| `batch_size_spots` | `int` | `4096` |
| `peak_chunk_size` | `int` | `4` |
| `eval_batch_size_spots` | `int` | `8192` |
| `global_learning_rate` | `float` | `0.01` |
| `local_learning_rate` | `float` | `0.01` |
| `global_hyperparameter_learning_rate` | `float` | `0.05` |
| `local_hyperparameter_learning_rate` | `float` | `0.02` |
| `global_initial_length_scale` | `float` | `160.0` |
| `local_initial_length_scale` | `float` | `20.0` |
| `local_min_length_scale` | `float` | `5.0` |
| `local_max_length_scale` | `float` | `40.0` |
| `local_neighbors` | `int` | `16` |
| `predecessor_candidate_multiplier` | `int` | `8` |
| `neighbor_query_batch_size` | `int` | `50000` |
| `geometry_cache_dir` | `str \| None` | `None` |
| `geometry_cache_mode` | `str` | `'off'` |
| `dtype` | `str` | `'float32'` |
| `device` | `str \| None` | `None` |
| `random_state` | `int` | `0` |
| `keep_component_outputs` | `bool` | `False` |
| `joint_epochs` | `int` | `5` |
| `global_min_length_scale` | `float` | `60.0` |
| `local_kl_batch_size` | `int` | `1024` |
| `local_kl_mode` | `LocalKLMode` | `'stochastic'` |
| `joint_global_variational_learning_rate` | `float` | `0.002` |
| `joint_global_hyperparameter_learning_rate` | `float` | `0.005` |
| `joint_local_variational_learning_rate` | `float` | `0.002` |
| `joint_local_hyperparameter_learning_rate` | `float` | `0.002` |
| `joint_likelihood_learning_rate` | `float` | `0.002` |
| `joint_noise_lower_bound` | `float` | `1e-05` |
| `joint_gradient_clip` | `float \| None` | `None` |
| `save_component_state_dicts` | `bool` | `False` |
| `early_stopping_patience` | `int` | `0` |
| `early_stopping_min_delta` | `float` | `0.0` |
| `local_pretrain_target` | `LocalPretrainTarget` | `'global_residual'` |
| `local_pretrain_cache_dir` | `str \| None` | `None` |
| `local_pretrain_cache_mode` | `LocalPretrainCacheMode` | `'off'` |
| `observation_model` | `ObservationModel` | `'gaussian'` |
| `global_grid_eigensolver` | `str` | `'torch'` |
| `global_reuse_identical_axes` | `bool` | `False` |
| `global_variational_covariance` | `str` | `'meanfield'` |
| `global_covariance_rank` | `int` | `0` |
| `global_covariance_initial_fraction` | `float` | `0.01` |
| `global_covariance_init_seed` | `int \| None` | `None` |
| `global_prior_jitter` | `float` | `0.001` |
| `global_pretrain_updates` | `int \| None` | `None` |
| `local_pretrain_updates` | `int \| None` | `None` |
| `joint_updates` | `int \| None` | `None` |
| `global_grid_phase` | `tuple[float, float]` | `(0.0, 0.0)` |
| `fixed_kernel_parameters` | `dict \| None` | `None` |
| `fixed_local_length_scale` | `bool` | `False` |

### `fit_gp`

Alias of fit_joint_global_local_gp_spatial_maps. Write native posterior-map artifacts and return a summary dictionary. Low-level fitting does not automatically reproduce the complete Workflow production adapter.

Fit additive global V-SKI and local VNNGP components jointly.

Sequential global and local fits initialize the model. The final stage uses
either the historical Gaussian observation model or an exposure-offset
Poisson log-link and a factorized variational family ``q(global) q(local)``.
Consequently, component covariance is zero under q and latent variances
add; this is not exact full-posterior inference.

[Implementation](../src/needlepoint/gpytorch_joint_global_local.py).

```python
ndl.fit_gp(
    expression: 'sparse.spmatrix | np.ndarray',
    coords: 'np.ndarray',
    feature_names: 'np.ndarray',
    output_dir: 'str | Path',
    *,
    settings: 'JointGlobalLocalGPSettings | None' = None,
    prediction_coords: 'np.ndarray | None' = None,
    prediction_spot_indices: 'np.ndarray | None' = None,
    training_prediction_rows: 'np.ndarray | None' = None,
    global_initial_length_scales: 'np.ndarray | None' = None,
    exposure: 'np.ndarray | None' = None,
    include_gaussian_latent_aliases: 'bool | None' = None,
) -> 'dict'
```

### `require_production_gp_artifacts`

Runtime validation raises for unsupported or incomplete production coverage.

Validate that all source map directories are production-ready.

[Implementation](../src/needlepoint/artifacts.py).

```python
ndl.require_production_gp_artifacts(
    paths: 'Iterable[str | Path]',
    *,
    allow_nonproduction: 'bool' = False,
    allow_legacy_full_coverage: 'bool' = False,
) -> 'list[GPArtifactAudit]'
```

## Prototypes and spatial domains

### `SpatialPrototypeSettings`

Settings shared by functional and Gaussian prototype clustering.

[Implementation](../src/needlepoint/spatial_prototypes.py).

| Parameter | Type | Default |
| --- | --- | --- |
| `method` | `str` | `'functional'` |
| `n_prototypes` | `int` | `20` |
| `max_iter` | `int` | `20` |
| `n_init` | `int` | `3` |
| `beta` | `float` | `10.0` |
| `tau2` | `float` | `0.1` |
| `update_tau2` | `bool` | `False` |
| `tau2_min` | `float` | `0.0001` |
| `tau2_max` | `float` | `10.0` |
| `gaussian_evidence_scale` | `float \| None` | `None` |
| `gaussian_evidence_power` | `float` | `1.0` |
| `mixture_weight_mode` | `str` | `'learned'` |
| `mixture_smoothing` | `float` | `1.0` |
| `responsibility_damping` | `float` | `1.0` |
| `tolerance` | `float` | `1e-05` |
| `spot_chunk_size` | `int` | `4096` |
| `preload_to_device` | `bool` | `True` |
| `initialization_spots` | `int` | `5000` |
| `initialization_smoothing` | `float` | `0.02` |
| `initialization_method` | `str` | `'spatial-sample'` |
| `initialization_embedding_dir` | `str \| None` | `None` |
| `merge_similar_prototypes` | `bool` | `True` |
| `merge_map_correlation_threshold` | `float` | `0.9` |
| `merge_membership_correlation_threshold` | `float` | `-1.0` |
| `merge_linkage` | `str` | `'complete'` |
| `min_prototypes_after_merge` | `int` | `2` |
| `device` | `str \| None` | `None` |
| `dtype` | `str` | `'float32'` |
| `random_state` | `int` | `0` |

### `fit_prototypes`

Alias of fit_spatial_prototypes. Consume a GPPosteriorFactorStore, write prototype artifacts, and return SpatialPrototypeResult. Existing GP fields are not refitted.

Fit peak-pattern prototypes without refitting the feature-specific GPs.

[Implementation](../src/needlepoint/spatial_prototypes.py).

```python
ndl.fit_prototypes(
    store: 'GPPosteriorFactorStore',
    output_dir: 'Path',
    settings: 'SpatialPrototypeSettings | None' = None,
) -> 'SpatialPrototypeResult'
```

### `merge_prototypes`

Alias of merge_spatial_prototype_directory. Merge a saved prototype package with explicit correlation criteria and retain the upstream optimizer identity.

Apply the conservative merge stage to an existing prototype fit.

[Implementation](../src/needlepoint/spatial_prototypes.py).

```python
ndl.merge_prototypes(
    source_dir: 'Path',
    output_dir: 'Path',
    *,
    map_correlation_threshold: 'float' = 0.9,
    membership_correlation_threshold: 'float' = -1.0,
    min_groups: 'int' = 2,
) -> 'SpatialPrototypeResult'
```

### `VSKIMapStore`

Read native posterior-map chunks with aligned feature and spot metadata; iter_blocks() streams active blocks.

Chunked reader for V-SKI posterior maps.

The store keeps V-SKI fitted maps on disk and streams only the requested
spot rows and peak columns. This is the central scalability constraint for
TEMI: downstream LBM updates operate on small summaries, while the
N-by-P posterior maps are read chunk by chunk.

[Implementation](../src/needlepoint/vski_lbm.py).

```python
ndl.VSKIMapStore(
    vski_dir: 'Path',
    *,
    max_spots: 'int | None' = None,
    max_peaks: 'int | None' = None,
    spot_indices: 'np.ndarray | None' = None,
    peak_positions: 'np.ndarray | None' = None,
    selected_models_file: 'Path | None' = None,
    map_standardization: 'str' = 'none',
    use_posterior_variance: 'bool' = True,
    require_production_fit: 'bool' = False,
    allow_nonproduction_fit: 'bool' = False,
    allow_legacy_full_coverage: 'bool' = False,
) -> 'None'
```

```python
VSKIMapStore.iter_blocks(
    self,
    *,
    include_variance: 'bool' = True,
) -> 'Iterator[tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]]'
```

### `VSKILBM`

Fixed-domain latent block model over a SpatialMapStore. fit() returns VSKILBMResult. Constructor defaults differ from the production Workflow adapter, which explicitly selects a boundary-aware graph and Torch backend.

Latent block model over fitted V-SKI posterior maps.

This model jointly updates peak-module responsibilities and spot-domain
responsibilities from fitted GP posterior means. Posterior variance enters
the block variance estimate, so uncertain fitted regions contribute less
decisive likelihood evidence. Spatial coherence is added with a sparse kNN
graph over spots and does not require dense GP covariance operations.

[Implementation](../src/needlepoint/vski_lbm.py).

```python
ndl.VSKILBM(
    *,
    n_peak_modules: 'int' = 20,
    n_domains: 'int' = 12,
    graph_neighbors: 'int' = 16,
    graph_construction: 'str' = 'auto',
    graph_radius: 'float | None' = None,
    graph_distance_bandwidth: 'float | None' = None,
    lambda_z: 'float' = 0.5,
    lambda_z_min: 'float' = 0.0,
    peak_prior_decay_tau: 'float' = 5.0,
    likelihood_power_a: 'float' = 1.0,
    likelihood_power_b: 'float' = 1.0,
    likelihood_scaling: 'str' = 'canonical',
    likelihood_power_coupling: 'str' = 'independent',
    shared_likelihood_power: 'float | None' = None,
    evidence_aggregation: 'str' = 'sum',
    likelihood_model: 'str' = 'block-average',
    residual_weight: 'float' = 0.0,
    residual_variance: 'str' = 'module',
    graph_model: 'str' = 'signed-roughness',
    beta: 'float' = 1.0,
    roughness_weight: 'float' = 0.25,
    graph_boundary_scale: 'float' = 1.0,
    graph_message_normalization: 'str' = 'sum',
    domain_prior_weight: 'float' = 0.0,
    domain_mixture_prior: 'str' = 'none',
    domain_dirichlet_alpha0: 'float' = 1.0,
    domain_mixture_prior_weight: 'float' = 1.0,
    active_domain_weight_threshold: 'float' = 0.005,
    active_domain_min_spots: 'float' = 1.0,
    domain_update: 'str' = 'auto',
    domain_capacity_factor: 'float' = 0.0,
    domain_size_penalty_weight: 'float' = 0.0,
    domain_hard_capacity_factor: 'float' = 0.0,
    domain_hard_smooth: 'float' = 0.02,
    capped_softmax_iter: 'int' = 50,
    capped_softmax_step: 'float' = 1.0,
    capped_softmax_tol: 'float' = 0.001,
    final_domain_labeling: 'str' = 'argmax',
    final_domain_capacity_factor: 'float' = 0.0,
    max_iter: 'int' = 30,
    tol: 'float' = 0.0001,
    variance_floor: 'float' = 0.0001,
    min_peak_module_mass: 'float' = 1.0,
    min_domain_mass: 'float' = 1.0,
    epsilon_min: 'float' = 0.05,
    epsilon_max: 'float' = 0.6,
    a_temperature: 'float' = 1.0,
    b_temperature: 'float' = 1.0,
    edge_batch_size: 'int' = 200000,
    sample_spots_for_init: 'int' = 60000,
    peak_init_n_init: 'int' = 20,
    factor_feature_prior_weight: 'float' = 0.0,
    factor_location_prior_weight: 'float' = 0.0,
    factor_feature_prior_temperature: 'float' = 1.0,
    factor_location_prior_temperature: 'float' = 1.0,
    prototype_transition_smoothing: 'float' = 1.0,
    prototype_transition_damping: 'float' = 0.5,
    update_prototype_transition: 'bool' = True,
    prototype_transition_identity_smoothing: 'float' = 0.05,
    prototype_peak_initialization_smoothing: 'float' = 0.05,
    prototype_general_initialization_blend: 'float' = 0.5,
    domain_initialization: 'str' = 'soft-kmeans-module-maps',
    domain_initialization_temperature: 'float | None' = None,
    domain_initialization_n_init: 'int' = 20,
    domain_initialization_module_weighting: 'str' = 'mass-reliability',
    feature_likelihood_weighting: 'str' = 'uniform',
    graph_module_weighting: 'str' = 'uniform',
    dataset_signature_model: 'str' = 'shared',
    dataset_signature_shrinkage_mass: 'float | None' = 10000.0,
    compute_backend: 'str' = 'numpy',
    compute_device: 'str' = 'cuda',
    compute_dtype: 'str' = 'float32',
    random_state: 'int' = 0,
    verbose: 'bool' = False,
) -> 'None'
```

```python
VSKILBM.fit(
    self,
    store: 'SpatialMapStore',
    *,
    peak_prior_labels: 'np.ndarray | None' = None,
    peak_prior_distance: 'np.ndarray | None' = None,
    initial_domain_labels: 'np.ndarray | None' = None,
    initial_peak_responsibilities: 'np.ndarray | None' = None,
    fixed_peak_prior_responsibilities: 'np.ndarray | None' = None,
    initial_domain_responsibilities: 'np.ndarray | None' = None,
    factor_feature_embedding: 'np.ndarray | None' = None,
    factor_location_embedding: 'np.ndarray | None' = None,
    prototype_responsibilities: 'np.ndarray | None' = None,
    precomputed_graph: 'sparse.spmatrix | None' = None,
    persistent_graph_module_weights: 'np.ndarray | None' = None,
) -> 'VSKILBMResult'
```

## Explicit non-Gaussian models

### `SpatialBounds`

XYXY bounds in micrometres: xmin, ymin, xmax, ymax.

Named fields avoid ambiguous four-tuples. Historical LGCP XXYY tuples
require the explicit ``from_xxyy`` conversion; their order is never guessed.

[Implementation](../src/needlepoint/bounds.py).

```python
ndl.SpatialBounds(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
) -> None
```

### `CoupledVISettings`

Experimental coupled Gaussian VI settings. Dense and PCG/Rao–Blackwell alternatives retain their numerical and convergence diagnostics.

[Implementation](../src/needlepoint/non_gaussian/vi.py).

| Parameter | Type | Default |
| --- | --- | --- |
| `backend` | `str` | `'dense'` |
| `variance_method` | `str` | `'exact'` |
| `max_iterations` | `int` | `200` |
| `stationarity_tolerance` | `float` | `1e-05` |
| `initial_step` | `float` | `1.0` |
| `minimum_step` | `float` | `5.960464477539063e-08` |
| `fit_baseline` | `bool` | `True` |
| `baseline_prior_sd` | `float \| None` | `None` |
| `fit_noise` | `bool` | `False` |
| `noise_bounds` | `tuple[float, float] \| None` | `None` |
| `dense_max_dimension` | `int` | `2048` |
| `rhs_batch_size` | `int` | `8` |
| `rb_block_size` | `int` | `64` |
| `training_draws` | `int` | `256` |
| `max_training_draws` | `int` | `4096` |
| `final_draws` | `int` | `2048` |
| `maximum_relative_mcse` | `float` | `0.05` |
| `maximum_exponential_bias_proxy` | `float` | `0.01` |
| `reference_relative_tolerance` | `float` | `0.001` |
| `logdet_method` | `str` | `'slq'` |
| `logdet_probes` | `int` | `16` |
| `lanczos_steps` | `int` | `40` |
| `pcg_rtol` | `float` | `1e-10` |
| `pcg_atol` | `float` | `1e-12` |
| `pcg_max_iterations` | `int` | `4000` |
| `residual_refresh` | `int` | `25` |
| `memory_cap_bytes` | `int` | `8589934592` |
| `seed` | `int` | `20261019` |
| `failure_policy` | `str` | `'raise'` |

### `fit_spatial_vi`

Yield (feature_name, CoupledVIResult) in panel order. Accept raw Poisson counts or Tobit continuous observations and a supplied prior. LGCP requires original events.

Yield (feature_name, fit) in panel order, fitting one feature at a time.

All common matrix readers can feed this iterator. Poisson consumes raw
counts; Tobit requires continuous measurements and explicit censor flags.
Feature-specific censor flags/thresholds must match the full input matrix
(or use a scalar/one value per location). No count normalization is applied.
LGCP uses read_point_pattern + fit_lgcp to preserve original events.

[Implementation](../src/needlepoint/non_gaussian/interface.py).

```python
ndl.fit_spatial_vi(
    data,
    prior,
    *,
    observation_model,
    features=None,
    settings=None,
    exposure=1.0,
    exposure_semantics='unit',
    is_censored=None,
    threshold=None,
    noise_sd=None,
    measurement_scale=None,
    read_options=None,
)
```

### `fit_tobit_gp`

Return CoupledVIResult. Explicit censor flags, threshold, Gaussian noise SD, and a measurement-scale description are required.

Fit zero-coded left-censoring with explicit flags and Gaussian noise.

[Implementation](../src/needlepoint/non_gaussian/vi.py).

```python
ndl.fit_tobit_gp(
    prior,
    values,
    *,
    is_censored,
    threshold,
    noise_sd,
    coordinates=None,
    evaluation=None,
    settings=None,
    quadrature_order=64,
    baseline=None,
    measurement_scale='declared_gaussian',
)
```

### `fit_poisson_gp`

Return CoupledVIResult from raw counts and explicit exposure semantics. This does not normalize counts.

Fit raw counts with all supplied rows, including acquired zeros.

[Implementation](../src/needlepoint/non_gaussian/vi.py).

```python
ndl.fit_poisson_gp(
    prior,
    counts,
    *,
    coordinates=None,
    evaluation=None,
    exposure=1.0,
    exposure_semantics='unit',
    settings=None,
    baseline=None,
)
```

### `fit_lgcp`

Return CoupledVIResult from original events and acquisition-window quadrature/effort. Binned counts do not substitute for event locations.

Fit original point events and a physical-area spatial integral.

[Implementation](../src/needlepoint/non_gaussian/vi.py).

```python
ndl.fit_lgcp(
    prior,
    events,
    quadrature,
    *,
    settings=None,
    baseline=None,
)
```

### `VIStorageSettings`

Export precision, field scale, and retention settings. Inference precision is configured separately.

Compact coupled-VI exports; numerical fitting and state stay unchanged.

[Implementation](../src/needlepoint/storage.py).

| Parameter | Type | Default |
| --- | --- | --- |
| `output_dtype` | `str` | `'float32'` |
| `field_scale` | `str` | `'latent_gp'` |
| `retain_component_maps` | `bool` | `False` |
| `retain_numerical_maps` | `bool` | `False` |
| `retain_observation_maps` | `bool` | `False` |
| `retain_state` | `bool` | `True` |

### `export_coupled_maps`

Return an output Path after writing a shared chunked map package. Field scale and units are required. Production export requires converged fits and an explicit complete-support declaration; observed sampling-noise moments are excluded from production prototype/LBM fields.

Stream fitted features into maps consumed by prototype and LBM readers.

``fits`` is a mapping or iterable of (name, CoupledVIResult). Features must
share the native evaluation coordinates, likelihood and eligible support.
The caller explicitly declares production intent and complete acquired
observation coverage; numerical convergence alone does not establish it.
No parameters are refitted and no posterior variance is recombined.

[Implementation](../src/needlepoint/non_gaussian/exports.py).

```python
ndl.export_coupled_maps(
    fits,
    output_directory,
    *,
    field_scale,
    field_units,
    fit_purpose='evaluation',
    complete_observation_support=False,
    output_dtype='float32',
)
```

## Observation-level spatial factors

### `SpatialFactorSettings`

Experimental observation-level factor settings. Geometry, kernels, device/dtype, and location batches use a separately supplied JointGlobalLocalGPSettings; feature-GP pretraining stages are not used.

Fixed-epoch, mean-field variational factor fitting.

``JointGlobalLocalGPSettings`` supplied to the fit configures geometry,
kernels, local KL sampling, device, dtype, and location batch sizes. Its
feature pretraining stages are not used by this joint factor model.

[Implementation](../src/needlepoint/spatial_factors/settings.py).

| Parameter | Type | Default |
| --- | --- | --- |
| `n_spatial_factors` | `int` | `8` |
| `n_nonspatial_factors` | `int` | `0` |
| `nonnegative` | `bool` | `True` |
| `epochs` | `int` | `100` |
| `learning_rate` | `float` | `0.01` |
| `hyperparameter_learning_rate` | `float` | `0.002` |
| `feature_batch_size` | `int` | `256` |
| `n_samples` | `int` | `4` |
| `loading_l1` | `float` | `0.0` |
| `gradient_clip` | `float` | `10.0` |
| `learn_kernel` | `bool` | `True` |
| `initialization` | `str` | `'nmf'` |
| `initialization_spots` | `int` | `10000` |
| `initialization_iterations` | `int` | `100` |
| `global_initialization` | `str` | `'kernel_smooth'` |
| `variational_scale_parameterization` | `str` | `'softplus'` |
| `nonspatial_prior_sd` | `float` | `1.0` |
| `learn_nonspatial_prior` | `bool` | `True` |
| `random_state` | `int` | `0` |

### `SpatialFactorResult`

Fitted factors/loadings, histories, and metadata. Also exposes factor means/variances, coordinates, feature mixtures, feature/observation spatial importance, predict(), and save().

[Implementation](../src/needlepoint/spatial_factors/fit.py).

```python
ndl.SpatialFactorResult(
    model: needlepoint.spatial_factors.model.GlobalLocalFactorModel,
    feature_names: numpy.ndarray,
    history: pandas.core.frame.DataFrame,
    arrays: dict,
    metadata: dict,
    exposure: needlepoint.spatial_factors.fit.AlignedArray,
    threshold: needlepoint.spatial_factors.fit.AlignedArray | None = None,
) -> None
```

```python
SpatialFactorResult.predict(
    self,
    coordinates=None,
    **kwargs,
)
```

```python
SpatialFactorResult.save(
    self,
    output_dir,
)
```

### `fit_spatial_factors`

Return SpatialFactorResult. Likelihood-specific exposure, masks, censoring, and measurement scale are explicit. This interface remains experimental.

Fit shared spatial/nonspatial factors directly to a matrix or SpatialData.

No normalization, standardization, feature filtering, or row removal occurs.
Counts must be raw integers, with known exposure supplied explicitly. A
Tobit fit additionally requires measurement_scale, flags, and thresholds.
Features are evaluated in bounded blocks, with shared sample gradients
accumulated before a single backward pass through the GP factor graph.
``epoch_callback(model, record)`` optionally observes each completed epoch
for progress logging or tensor checkpointing; it must not mutate the model.

[Implementation](../src/needlepoint/spatial_factors/fit.py).

```python
ndl.fit_spatial_factors(
    values,
    coordinates=None,
    *,
    likelihood='poisson',
    settings=None,
    gp_settings=None,
    feature_names=None,
    exposure=1.0,
    exposure_semantics='unit',
    is_censored=None,
    threshold=None,
    measurement_scale=None,
    output_dir=None,
    epoch_callback=None,
)
```

### `load_spatial_factors`

Return SpatialFactorResult from saved state. Saved parameterization metadata governs historical replay.

Verify file digests and rebuild a saved factor model without raw data.

[Implementation](../src/needlepoint/spatial_factors/fit.py).

```python
ndl.load_spatial_factors(
    output_dir,
    *,
    device='cpu',
    likelihood=None,
)
```

## Additional module interfaces

Reusable modules under `needlepoint.*` provide the interfaces listed below.
Optional numerical methods retain their experimental designation.

| Module | Utility |
| --- | --- |
| [`needlepoint.scoring.multiscale_contrast`](../src/needlepoint/scoring/multiscale_contrast.py) | Multiscale scoring and empirical FDR |
| [`needlepoint.gp_initialization`](../src/needlepoint/gp_initialization.py) | Length-scale policies and screening |
| [`needlepoint.gp_frozen_state`](../src/needlepoint/gp_frozen_state.py) | Frozen fitted-prior snapshots and identities |
| [`needlepoint.gp_joint_conditioning`](../src/needlepoint/gp_joint_conditioning.py) | Frozen Gaussian joint conditioning |
| [`needlepoint.gp_prior_operators`](../src/needlepoint/gp_prior_operators.py) | Matrix-free structured prior actions |
| [`needlepoint.gp_variance_estimation`](../src/needlepoint/gp_variance_estimation.py) | Streaming posterior moments |
| [`needlepoint.gp_covariance`](../src/needlepoint/gp_covariance.py) | Opt-in low-rank inducing covariance |
| [`needlepoint.non_gaussian.data`](../src/needlepoint/non_gaussian/data.py) | read_point_pattern and regular_grid_quadrature |
| [`needlepoint.non_gaussian.operators`](../src/needlepoint/non_gaussian/operators.py) | DensePrior, StructuredSpatialPrior, build_spatial_prior |
| [`needlepoint.non_gaussian.likelihoods`](../src/needlepoint/non_gaussian/likelihoods.py) | Tobit, Poisson, and LGCP adapters |
| [`needlepoint.non_gaussian.vi`](../src/needlepoint/non_gaussian/vi.py) | CoupledVIResult, predictions, convergence, state export |
| [`needlepoint.posterior_blocks`](../src/needlepoint/posterior_blocks.py) | Native map-block access |
| [`needlepoint.shared_spatial_factorization`](../src/needlepoint/shared_spatial_factorization.py) | GPPosteriorFactorStore and shared field factors |
| [`needlepoint.multiscale_prototypes`](../src/needlepoint/multiscale_prototypes.py) | Optional multiscale prototype initialization |
| [`needlepoint.prototype_graph_coupling`](../src/needlepoint/prototype_graph_coupling.py) | Prototype graphs and cross-section coupling |
| [`needlepoint.spatial_factors.__init__`](../src/needlepoint/spatial_factors/__init__.py) | Observation-level spatial factors |
| [`needlepoint.factor_lbm_bridge`](../src/needlepoint/factor_lbm_bridge.py) | Factor-to-LBM input bridge |
| [`needlepoint.multidataset_lbm`](../src/needlepoint/multidataset_lbm.py) | Aligned and dataset-balanced multi-section stores |
| [`needlepoint.shared_private_lbm`](../src/needlepoint/shared_private_lbm.py) | Optional conditional shared/private LBM |
| [`needlepoint.spatial_unmixing`](../src/needlepoint/spatial_unmixing.py) | Gaussian spatial mixture factorization |
| [`needlepoint.graph`](../src/needlepoint/graph.py) | Spatial graph construction |
| [`needlepoint.domain_peak_statistics`](../src/needlepoint/domain_peak_statistics.py) | Streamed domain-by-feature summaries |
| [`needlepoint.domain_peak_differential`](../src/needlepoint/domain_peak_differential.py) | Domain-conditioned feature contrasts |
| [`needlepoint.storage`](../src/needlepoint/storage.py) | Export precision, retention, and runtime map contracts |
| [`needlepoint.artifacts`](../src/needlepoint/artifacts.py) | Runtime production coverage and artifact validation |
| [`needlepoint.provenance`](../src/needlepoint/provenance.py) | Runtime source/environment identities |

## Command-line interface

All commands accept `--help`. Stage inputs and outputs are caller supplied.
`needlepoint run` prints a plan until `--execute` is set.

| Unified command | Dedicated executable | Implementation | Purpose |
| --- | --- | --- | --- |
| `needlepoint run` | `needlepoint-run` | [Source](../src/needlepoint/cli/run_workflow.py) | Plan/execute/resume the count workflow |
| `needlepoint score` | `needlepoint-score` | [Source](../src/needlepoint/scoring/multiscale_contrast.py) | Score and select features |
| `needlepoint fit --mode screen` | `needlepoint-fit-h5ad --mode screen` | [Source](../src/needlepoint/cli/run_h5ad_joint_global_local.py) | Default Gaussian scale screening |
| `needlepoint fit --mode fit` | `needlepoint-fit-h5ad --mode fit` | [Source](../src/needlepoint/cli/run_h5ad_joint_global_local.py) | Gaussian or optional Poisson fitting |
| `needlepoint screen` | `needlepoint-screen-direct-ski` | [Source](../src/needlepoint/cli/run_direct_ski_screen.py) | Optional deterministic direct-SKI screen |
| `needlepoint fit-non-gaussian` | `needlepoint-fit-non-gaussian` | [Source](../src/needlepoint/cli/fit_non_gaussian.py) | One-feature Tobit/Poisson/LGCP fit |
| `needlepoint prototypes` | `needlepoint-prototypes` | [Source](../src/needlepoint/cli/run_spatial_prototypes.py) | Fit/merge spatial prototypes |
| `needlepoint lbm` | `needlepoint-lbm` | [Source](../src/needlepoint/cli/run_lbm.py) | Infer spatial domains |
| `needlepoint merge-gp-shards` | `needlepoint-merge-gp-shards` | [Source](../src/needlepoint/cli/merge_gp_shards.py) | Merge disjoint feature-map shards |
| `needlepoint merge-screens` | `needlepoint-merge-lengthscale-screens` | [Source](../src/needlepoint/cli/merge_lengthscale_screens.py) | Merge disjoint screen shards |
| `needlepoint provenance` | `needlepoint-provenance` | [Source](../src/needlepoint/cli/write_provenance.py) | Write source/environment identity |
| — | `needlepoint-condition-gaussian` | [Source](../src/needlepoint/cli/condition_gaussian.py) | Optional frozen Gaussian conditioning |
| `needlepoint defaults` | — | [Source](../src/needlepoint/__main__.py) | Print descriptive reference policy |

The standalone fit adapter defaults to evaluation holdouts. For downstream
production artifacts, set `--fit-purpose production`,
`--random-holdout-fraction 0`, and `--blocked-holdout-fraction 0`.
Workflow sets these options explicitly. Device and grid defaults are
stage-specific; inspect command help before independent runs.
