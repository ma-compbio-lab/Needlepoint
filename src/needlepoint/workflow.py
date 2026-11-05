"""A reviewable, resumable workflow built from the canonical stage adapters."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import shutil
import sys
import time

from .config import config_sha256
from .inputs import SpatialData, read_spatial
from .provenance import file_sha256, write_json_atomic
from .resources import Resources


@dataclass(frozen=True)
class WorkflowConfig:
    """Sequencing-count workflow policy; physical scales are micrometres.

    Explicit selected_genes bypasses discovery scoring for matched-panel
    experiments. Resource choices never truncate an FDR-selected panel.
    """
    selected_genes: tuple[str, ...] | None = None
    stop_after: str = 'lbm'
    grid_size: int = 128
    bandwidths_um: tuple[float, ...] = (20., 40., 80., 160., 320., 640.)
    target_sum: float = 10000.
    fdr_q: float = 0.05
    max_artifact_like_score: float = .35
    min_spots: int = 20
    minimum_selected_genes: int = 20
    null_profiles_per_stratum: int = 2000
    global_epochs: int = 30
    local_epochs: int = 30
    joint_epochs: int = 10
    global_min_length_scale: float = 40.
    global_initial_length_scale: float = 80.
    local_min_length_scale: float = 8.
    local_initial_length_scale: float = 16.
    local_max_length_scale: float = 32.
    local_neighbors: int = 16
    n_prototypes: int = 20
    prototype_iterations: int = 20
    prototype_n_init: int = 3
    prototype_merge_correlation: float = .90
    domain_counts: tuple[int, ...] = (4, 8, 12)
    lbm_iterations: int = 30
    random_state: int = 0
    retain_caches: bool = False
    input_storage: str = 'copy'
    length_scale_policy: str = 'physical_multiscale_v1'
    sampling_interval_um: float | None = None
    physical_local_floor_um: float = 10.
    physical_global_to_local_ratio: float = 4.
    physical_upper_fov_fraction: float = .5
    physical_max_candidates: int = 8
    observation_model: str = 'gaussian'

    def validate(self):
        import numpy as np
        if self.observation_model not in ('gaussian', 'poisson-log'):
            raise ValueError('observation_model must be gaussian or poisson-log')
        if self.input_storage not in ('copy', 'reference') or not isinstance(self.retain_caches, bool):
            raise ValueError('input_storage must be copy/reference; retain_caches must be boolean')
        if self.stop_after not in ('score', 'screen', 'gp', 'prototypes', 'lbm'):
            raise ValueError('stop_after must be score, screen, gp, prototypes, or lbm')
        if self.grid_size < 4 or min(self.min_spots, self.minimum_selected_genes,
             self.null_profiles_per_stratum, self.global_epochs, self.local_epochs,
             self.joint_epochs, self.local_neighbors, self.n_prototypes,
             self.prototype_iterations, self.prototype_n_init, self.lbm_iterations) < 1:
            raise ValueError('Grid >= 4 and positive counts/iteration limits required')
        integers = [self.grid_size,self.min_spots,self.minimum_selected_genes,self.null_profiles_per_stratum,
                    self.global_epochs,self.local_epochs,self.joint_epochs,self.local_neighbors,
                    self.n_prototypes,self.prototype_iterations,self.prototype_n_init,self.lbm_iterations]
        if any(int(v) != v for v in integers) or int(self.random_state) != self.random_state or self.random_state < 0:
            raise ValueError('Iteration limits, dimensions, and random seed must be integers; seed >= 0')
        scales = [*self.bandwidths_um, self.target_sum, self.global_min_length_scale, self.global_initial_length_scale,
                  self.local_min_length_scale, self.local_initial_length_scale, self.local_max_length_scale]
        if not self.bandwidths_um or not np.isfinite(scales).all() or min(scales) <= 0:
            raise ValueError('Finite positive physical scales and normalization target required')
        if not 0 < self.fdr_q <= 1 or not -1 <= self.prototype_merge_correlation <= 1:
            raise ValueError('Invalid FDR or prototype correlation threshold')
        if not 0 <= self.max_artifact_like_score <= 1:
            raise ValueError('Artifact-like score threshold must lie in [0,1]')
        if self.length_scale_policy not in ('physical_multiscale_v1', 'grid_aware_log2'):
            raise ValueError('Unsupported length_scale_policy')
        physical_values = [self.physical_local_floor_um, self.physical_global_to_local_ratio,
                           self.physical_upper_fov_fraction]
        if not np.isfinite(physical_values).all() or min(physical_values) <= 0:
            raise ValueError('Physical length-scale settings must be finite and positive')
        if self.physical_max_candidates < 2 or int(self.physical_max_candidates) != self.physical_max_candidates:
            raise ValueError('physical_max_candidates must be an integer >= 2')
        if self.sampling_interval_um is not None and (
            not np.isfinite(self.sampling_interval_um) or self.sampling_interval_um <= 0
        ):
            raise ValueError('sampling_interval_um must be finite and positive')
        if self.selected_genes is None and self.null_profiles_per_stratum < 2000:
            raise ValueError('Empirical-FDR scoring requires at least 2000 null profiles per stratum')
        if self.length_scale_policy == 'grid_aware_log2':
            if not self.local_min_length_scale < self.local_initial_length_scale < self.local_max_length_scale:
                raise ValueError('Local initial scale must lie strictly inside its bounds')
            if self.global_min_length_scale >= self.global_initial_length_scale:
                raise ValueError('Global initial scale must be above its lower bound')
        if not self.domain_counts or any(k < 2 or int(k) != k for k in self.domain_counts) or len(set(self.domain_counts)) != len(self.domain_counts):
            raise ValueError('Supply distinct domain counts >= 2')


@dataclass(frozen=True)
class WorkflowResult:
    output_dir: Path
    state: dict

    @property
    def gp_dir(self):
        return self.output_dir/'gp/fitted_maps'

    @property
    def prototype_dir(self):
        return self.output_dir/'prototypes'

    def domain_dir(self, n_domains):
        return self.output_dir/'lbm'/f'R{n_domains}'


class Workflow:
    """Plan or run discovery → screen → full-support GP → prototypes → LBM.

    Each stage runs in a separate process using the installed package. This
    avoids global argv changes and lets callers inspect commands and logs.
    This initial executor supports one visible GPU or CPU; scheduler allocation
    stays under caller control. Ingestion currently loads the sparse matrix in
    RAM. Lower-level module APIs remain available for other measurement types.
    """
    def __init__(self, data, output_dir, *, config=None, resources=None, **read_options):
        self.config = config or WorkflowConfig()
        self.config.validate()
        self.input_source = None
        self.input_source_sha256 = None
        if self.config.input_storage == 'reference':
            import anndata as ad
            if not isinstance(data, (str, Path)) or Path(data).suffix != '.h5ad':
                raise ValueError('Reference input requires an original canonical H5AD path')
            if (read_options.get('layer') is not None or read_options.get('coordinate_scale', 1) != 1
                    or read_options.get('coord_key', 'auto') not in ('auto', 'spatial')
                    or any(read_options.get(k) is not None for k in ('coords', 'genes', 'obs'))):
                raise ValueError('Reference input requires raw counts in X and spatial XY micrometres without conversion')
            self.input_source = Path(data).resolve()
            self.input_source_sha256 = file_sha256(self.input_source)
            source = ad.read_h5ad(self.input_source, backed='r')
            try:
                if 'spatial' not in source.obsm or source.obsm['spatial'].shape[1] != 2:
                    raise ValueError('Reference input requires canonical obsm[spatial] XY coordinates')
            finally:
                source.file.close()
        self.data = read_spatial(data, **read_options)
        self.input_source_data_sha256 = self.data.fingerprint() if self.input_source is not None else None
        if self.input_source is not None and file_sha256(self.input_source) != self.input_source_sha256:
            raise ValueError('Referenced input changed during ingestion')
        self.output_dir = Path(output_dir).resolve()
        self.resources = resources or Resources()
        if self.data.measurement != 'counts':
            raise ValueError('Workflow currently requires counts; continuous measurements use fit_gp directly')
        if self.config.stop_after != 'score':
            import numpy as np
            if np.any(np.ptp(self.data.coords,axis=0) <= 0):
                raise ValueError('GP workflow requires positive physical span on both coordinate axes')
        if self.data.shape[0] <= self.config.local_neighbors and self.config.stop_after not in ('score','screen'):
            raise ValueError('local_neighbors must be smaller than the observation count')
        if self.config.selected_genes is not None:
            genes = self.config.selected_genes
            if not genes or len(set(genes)) != len(genes) or not set(genes) <= set(self.data.genes):
                raise ValueError('Fixed gene panel must be nonempty, unique, and fully present')
            if self.config.stop_after in ('prototypes','lbm') and self.config.n_prototypes > len(genes):
                raise ValueError('n_prototypes exceeds the fixed gene panel; choose it explicitly')
        if self.config.stop_after == 'lbm' and max(self.config.domain_counts) > self.data.shape[0]:
            raise ValueError('Domain counts exceed observation count')

    def _source_hash(self, root=None):
        root = Path(__file__).parent if root is None else Path(root)
        h = hashlib.sha256()
        for p in sorted(root.rglob('*')):
            if not p.is_file() or p.suffix not in ('.py','.toml'): continue
            h.update(str(p.relative_to(root)).encode()); h.update(p.read_bytes())
        return h.hexdigest()

    def plan(self):
        c, root = self.config, self.output_dir
        data_sha256 = self.data.fingerprint()
        if self.input_source is not None and file_sha256(self.input_source) != self.input_source_sha256:
            raise ValueError('Referenced input changed; construct a new Workflow')
        if self.input_source is not None and data_sha256 != self.input_source_data_sha256:
            raise ValueError('Referenced input measurements/metadata changed in memory')
        allocation = self.resources.resolve(n_spots=self.data.shape[0],
            n_features=len(c.selected_genes) if c.selected_genes is not None else self.data.shape[1], grid_size=c.grid_size)
        data_path = root/'input.h5ad'
        panel = root/('selected_genes.csv' if c.selected_genes is not None else 'scoring/selected_genes.csv')
        stages = []
        def add(name, module, out, marker, args):
            stages.append({'name':name, 'output_dir':str(out), 'marker':str(out/marker),
                'command':[sys.executable, '-m', 'needlepoint.'+module, *map(str,args)]})
        device = allocation['device']
        if c.selected_genes is None:
            args = ['--h5ad',data_path,'--output-dir',root/'scoring','--coord-key','spatial',
                    '--target-sum',c.target_sum,'--transform','log1p','--selection-mode','empirical-fdr',
                    '--n-candidate-genes',0,'--min-spots',c.min_spots,'--fdr-q',c.fdr_q,
                    '--max-artifact-like-score',c.max_artifact_like_score,
                    '--minimum-selected-genes',c.minimum_selected_genes,
                    '--null-profiles-per-stratum',c.null_profiles_per_stratum,
                    '--chunk-size-genes',allocation['feature_chunk_size'],
                    '--fft-workers',allocation['cpu_threads'],'--plot-top-n',0,
                    '--permutation-seed',c.random_state,
                    '--smoothing-backend','grid-fft-cuda' if device.startswith('cuda') else 'auto',
                    '--cuda-device',device,'--allow-irregular-scatter','--bandwidths-um',*c.bandwidths_um]
            add('score','scoring.multiscale_contrast',root/'scoring','multiscale_gene_contrast_summary.json',args)
        common = ['--h5ad',data_path,'--selected-genes',panel,'--coord-key','spatial',
                  '--fit-purpose','production','--random-holdout-fraction',0,'--blocked-holdout-fraction',0,
                  '--target-sum',c.target_sum,'--grid-size',c.grid_size,
                   '--device',device,'--random-state',c.random_state,
                   '--length-scale-policy',c.length_scale_policy,
                   '--global-min-length-scale',c.global_min_length_scale,
                   '--physical-local-floor-um',c.physical_local_floor_um,
                   '--physical-global-to-local-ratio',c.physical_global_to_local_ratio,
                   '--physical-upper-fov-fraction',c.physical_upper_fov_fraction,
                   '--physical-max-candidates',c.physical_max_candidates]
        if c.sampling_interval_um is not None:
            common.extend(['--sampling-interval-um',c.sampling_interval_um])
        add('screen','cli.run_h5ad_joint_global_local',root/'screen','initial_length_scale_screen_summary.json',
            [*common,'--mode','screen','--output-dir',root/'screen',
             '--observation-model',c.observation_model,
             '--normalization-scope','full-library','--transform','log1p',
             '--screen-feature-chunk-size',allocation['feature_chunk_size'],'--no-write-holdout-table'])
        add('gp','cli.run_h5ad_joint_global_local',root/'gp','gp_fidelity_benchmark_summary.json',
            [*common,'--mode','fit','--output-dir',root/'gp',
             '--observation-model',c.observation_model,
             '--normalization-scope','exposure-offset' if c.observation_model == 'poisson-log' else 'full-library',
             '--transform','none' if c.observation_model == 'poisson-log' else 'log1p',
             '--initial-length-scale-table',root/'screen/initial_length_scale_scores.csv',
             '--global-epochs',c.global_epochs,'--local-epochs',c.local_epochs,'--joint-epochs',c.joint_epochs,
             '--global-initial-length-scale',c.global_initial_length_scale,
             '--local-min-length-scale',c.local_min_length_scale,'--local-initial-length-scale',c.local_initial_length_scale,
             '--local-max-length-scale',c.local_max_length_scale,'--local-neighbors',c.local_neighbors,
             '--peak-chunk-size',allocation['feature_chunk_size'],'--batch-size-spots',allocation['batch_size'],
             '--eval-batch-size-spots',allocation['batch_size'],'--local-kl-batch-size',allocation['local_kl_batch_size'],
             '--vnngp-geometry-cache-dir',root/'caches/geometry','--vnngp-geometry-cache-mode','readwrite'])
        add('prototypes','cli.run_spatial_prototypes',root/'prototypes','prototype_manifest.json',
            ['--vski-dir',root/'gp/fitted_maps','--output-dir',root/'prototypes',
             '--n-prototypes',c.n_prototypes,'--max-iter',c.prototype_iterations,'--n-init',c.prototype_n_init,
             '--merge-map-correlation-threshold',c.prototype_merge_correlation,
             '--spot-chunk-size',allocation['batch_size'],'--no-preload-to-device',
             '--device',device,'--random-state',c.random_state])
        for k in c.domain_counts:
            add(f'lbm_R{k}','cli.run_lbm',root/'lbm'/f'R{k}','vski_lbm_manifest.json',
                ['--vski-dir',root/'gp/fitted_maps','--output-dir',root/'lbm'/f'R{k}',
                 '--prototype-dir',root/'prototypes','--n-domains',k,'--max-iter',c.lbm_iterations,
                 '--compute-backend','torch','--compute-device',device,'--graph-model','boundary-aware',
                 '--random-state',c.random_state])
        order = ['score','screen','gp','prototypes','lbm']
        limit = order.index(c.stop_after)
        stages = [s for s in stages if order.index('lbm' if s['name'].startswith('lbm_') else s['name']) <= limit]
        versions = {}
        for name in ('numpy','scipy','anndata','pandas','scikit-learn','torch','gpytorch','linear-operator',
                     'h5py','pyarrow','matplotlib','threadpoolctl','zarr'):
            try: versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError: versions[name] = None
        identity = {'input_sha256': data_sha256, 'config':asdict(c),
                    'referenced_input':None if self.input_source is None else
                        dict(path=str(self.input_source), sha256=self.input_source_sha256),
                    'environment':{'python':sys.version.split()[0], 'packages':versions},
                    'resources':asdict(self.resources), 'source_sha256':self._source_hash()}
        return {**identity,'plan_sha256':config_sha256(identity),'shape':list(self.data.shape),
                'allocation':allocation,'output_dir':str(root),'stages':stages,
                'estimated_posterior_bytes':int(self.data.shape[0]*(len(c.selected_genes) if c.selected_genes is not None else self.data.shape[1])*4),
                'posterior_estimate_note':'Two FP16 maps; excludes metadata, temporary files and downstream outputs.'}

    def run(self, *, resume=False):
        """Execute the reviewed plan; resume only matching, verified stages.

        Failure logs and incomplete stage directories are preserved. Explicit
        resume moves an incomplete owned stage aside before retrying it.
        """
        import fcntl
        import os
        plan = self.plan()
        root = self.output_dir
        root.mkdir(parents=True,exist_ok=True)
        with (root/'.workflow.lock').open('a') as lock:
            try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError: raise RuntimeError('Workflow is already running in this output directory') from None
            state_path = root/'workflow_state.json'
            if state_path.exists():
                if not resume: raise FileExistsError('Existing workflow; use resume=True')
                state = json.loads(state_path.read_text())
                if state['plan_sha256'] != plan['plan_sha256']:
                    raise ValueError('Input, configuration, resources, environment or source changed; choose a new output directory')
                if file_sha256(root/'input.h5ad') != state['input_file_sha256']:
                    raise ValueError('Persisted input changed; cannot resume')
                if state.get('fixed_panel_sha256') and file_sha256(root/'selected_genes.csv') != state['fixed_panel_sha256']:
                    raise ValueError('Persisted gene panel changed; cannot resume')
                if self._source_hash(root/'source/needlepoint') != plan['source_sha256']:
                    raise ValueError('Frozen workflow source changed or is missing; cannot resume')
            else:
                if any(p.name != '.workflow.lock' for p in root.iterdir()):
                    raise FileExistsError('Refusing an existing nonempty output directory')
                if self.input_source is None:
                    self.data.write_h5ad(root/'input.h5ad')
                else:
                    (root/'input.h5ad').symlink_to(self.input_source)
                shutil.copytree(Path(__file__).parent, root/'source/needlepoint',
                    ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
                if self._source_hash(root/'source/needlepoint') != plan['source_sha256']:
                    raise RuntimeError('Package source changed while freezing workflow')
                if self.config.selected_genes is not None:
                    import pandas as pd
                    pd.DataFrame({'gene':self.config.selected_genes}).to_csv(root/'selected_genes.csv',index=False)
                write_json_atomic(root/'workflow_plan.json',plan)
                state = {'plan_sha256':plan['plan_sha256'],'status':'running','stages':{},
                         'input_file_sha256':file_sha256(root/'input.h5ad')}
                if self.config.selected_genes is not None:
                    state['fixed_panel_sha256'] = file_sha256(root/'selected_genes.csv')
            env = os.environ.copy()
            env['PYTHONPATH'] = str(root/'source')
            threads = str(plan['allocation']['cpu_threads'])
            env.update(OMP_NUM_THREADS=threads,MKL_NUM_THREADS=threads,OPENBLAS_NUM_THREADS=threads,
                       MPLBACKEND='Agg')
            for stage in plan['stages']:
                if self.input_source is not None and file_sha256(root/'input.h5ad') != state['input_file_sha256']:
                    raise ValueError('Referenced input changed; cannot continue')
                name = stage['name']; marker = Path(stage['marker'])
                prior = state['stages'].get(name)
                if prior and prior['status']=='complete':
                    for path, digest in prior['files'].items():
                        if not Path(path).is_file() or file_sha256(path) != digest:
                            raise ValueError(f'Completed {name} artifact changed or missing: {path}')
                    continue
                out = Path(stage['output_dir'])
                if out.exists():
                    if not resume: raise FileExistsError(out)
                    out.rename(out.with_name(out.name+f'.incomplete_{time.time_ns()}'))
                state['status']='running'; state['stages'][name]={'status':'running','started':time.time(),
                    'allocation':plan['allocation'],'command':stage['command']}
                write_json_atomic(state_path,state,overwrite=True)
                (root/'logs').mkdir(exist_ok=True)
                print(f'Needlepoint: {name} on {plan["allocation"]["device"]}',flush=True)
                try:
                    with (root/'logs'/f'{name}.log').open('a') as log:
                        subprocess.run(stage['command'],stdout=log,stderr=subprocess.STDOUT,env=env,check=True)
                    if not marker.is_file(): raise RuntimeError(f'{name} exited without its completion marker')
                    if name == 'gp':
                        from .artifacts import require_production_gp_artifacts
                        require_production_gp_artifacts([root/'gp/fitted_maps'])
                        from .storage import cleanup_public_workflow_cache
                        cleanup = cleanup_public_workflow_cache(root, retain_caches=self.config.retain_caches)
                        state['stages'][name]['storage_cleanup'] = cleanup
                    # Hash every stage file to detect stale/modified arrays on
                    # resume, rather than trusting a completion JSON alone.
                    files = {str(p):file_sha256(p) for p in sorted(out.rglob('*')) if p.is_file()}
                    state['stages'][name].update(status='complete',files=files,finished=time.time())
                except Exception as e:
                    state['status']='failed'; state['stages'][name].update(status='failed',error=str(e))
                    write_json_atomic(state_path,state,overwrite=True)
                    raise RuntimeError(f'{name} failed; see {root/"logs"/(name+".log")}') from e
                write_json_atomic(state_path,state,overwrite=True)
            state['status']='complete'
            write_json_atomic(state_path,state,overwrite=True)
        return WorkflowResult(root,state)
