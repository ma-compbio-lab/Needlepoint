"""Prepare the opt-in full-support multiscale prototype initialization cache."""
import argparse
import json
import os
import shutil
from pathlib import Path

from needlepoint.shared_spatial_factorization import GPPosteriorFactorStore
from needlepoint.multiscale_prototypes import MultiscaleSketchSettings, prepare_multiscale_sketch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vski-dir', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--local-cache-dir', type=Path, default=None,
                        help='Build disk-backed arrays on local scratch, then publish one sequential copy.')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--pitch-um', type=float, required=True)
    parser.add_argument('--feature-chunk-size', type=int, default=32)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    store = GPPosteriorFactorStore(args.vski_dir, args.local_cache_dir or args.cache_dir, normalization_mode='shape',
        nonnegative_projection='rectified_gaussian', require_production_fit=True)
    if args.local_cache_dir is not None:
        report=store.prepare()
        if args.cache_dir.exists():
            raise FileExistsError(f'Refusing existing published cache {args.cache_dir}')
        args.cache_dir.parent.mkdir(parents=True,exist_ok=True)
        staging=args.cache_dir.with_name(args.cache_dir.name+f'.copying-{os.getpid()}')
        shutil.copytree(args.local_cache_dir,staging)
        check=GPPosteriorFactorStore(args.vski_dir,staging,normalization_mode='shape',
            nonnegative_projection='rectified_gaussian',require_production_fit=True)
        if check.prepare()['source_signature']!=report['source_signature']:
            raise ValueError('Published cache source mismatch')
        staging.rename(args.cache_dir)
        print(json.dumps(dict(stage='cache_published',local_cache=str(args.local_cache_dir),
            published_cache=str(args.cache_dir))),flush=True)
    settings = MultiscaleSketchSettings(pitch_um=args.pitch_um,
        feature_chunk_size=args.feature_chunk_size, device=args.device)
    print(json.dumps(prepare_multiscale_sketch(store, args.output_dir, settings), indent=2), flush=True)


if __name__ == '__main__':
    main()
