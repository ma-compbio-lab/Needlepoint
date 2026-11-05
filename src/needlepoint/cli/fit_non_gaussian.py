"""Explicit one-feature experimental Tobit/Poisson/LGCP fitting command."""
import argparse


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, choices=['tobit', 'poisson', 'lgcp'])
    parser.add_argument('--input', required=True, help='NPZ with coordinates and model-specific raw observations')
    parser.add_argument('--output', required=True, help='New output directory')
    parser.add_argument('--global-lengthscale', required=True, type=float, help='Micrometers')
    parser.add_argument('--local-lengthscale', required=True, type=float, help='Micrometers')
    parser.add_argument('--global-scale', type=float, default=1.0)
    parser.add_argument('--local-scale', type=float, default=1.0)
    parser.add_argument('--grid-size', type=int, default=16)
    parser.add_argument('--neighbors', type=int, default=16)
    parser.add_argument('--noise-sd', type=float, help='Required Gaussian measurement SD for Tobit')
    parser.add_argument('--measurement-scale', help='Required Gaussian measurement-scale description for Tobit')
    parser.add_argument('--exposure-semantics', default='unit')
    parser.add_argument('--baseline', type=float)
    parser.add_argument('--fix-baseline', action='store_true')
    parser.add_argument('--baseline-prior-sd', type=float)
    parser.add_argument('--backend', choices=['dense', 'pcg'], default='dense')
    parser.add_argument('--variance-method', choices=['exact', 'rb'], default='exact')
    parser.add_argument('--logdet-method', choices=['dense', 'slq'], default='slq')
    parser.add_argument('--max-iterations', type=int, default=200)
    parser.add_argument('--stationarity-tolerance', type=float, default=1e-5)
    parser.add_argument('--training-draws', type=int, default=256)
    parser.add_argument('--final-draws', type=int, default=2048)
    parser.add_argument('--rb-block-size', type=int, default=64)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=20261019)
    parser.add_argument('--bounds-order', choices=['xyxy', 'xxyy'], default='xyxy',
                        help='Acquisition bounds order; xxyy explicitly converts historical LGCP inputs')
    parser.add_argument('--output-dtype', choices=['float16', 'float32', 'float64'], default='float32')
    parser.add_argument('--field-scale', choices=['latent_gp', 'latent_predictor', 'rate', 'observed'], default='latent_gp')
    parser.add_argument('--retain-component-maps', action='store_true')
    parser.add_argument('--retain-numerical-maps', action='store_true')
    parser.add_argument('--retain-observation-maps', action='store_true')
    parser.add_argument('--retain-state', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if args.model == 'tobit' and (args.noise_sd is None or args.measurement_scale is None):
        parser.error('Tobit requires --noise-sd and --measurement-scale')
    import json
    import numpy as np
    from needlepoint.non_gaussian import (CoupledVISettings, build_spatial_prior,
        fit_tobit_gp, fit_poisson_gp, fit_lgcp, PointPattern, Quadrature)
    with np.load(args.input, allow_pickle=False) as arrays:
        data = {name: arrays[name].copy() for name in arrays.files}
    prior = build_spatial_prior(data['coordinates'], global_lengthscale=args.global_lengthscale,
        local_lengthscale=args.local_lengthscale, global_scale=args.global_scale,
        local_scale=args.local_scale, grid_shape=(args.grid_size, args.grid_size),
        local_neighbors=args.neighbors, device=args.device)
    settings = CoupledVISettings(backend=args.backend, variance_method=args.variance_method,
        logdet_method=args.logdet_method, max_iterations=args.max_iterations,
        stationarity_tolerance=args.stationarity_tolerance, fit_baseline=not args.fix_baseline,
        baseline_prior_sd=args.baseline_prior_sd, training_draws=args.training_draws,
        max_training_draws=max(4096, args.training_draws), final_draws=args.final_draws,
        rb_block_size=args.rb_block_size, seed=args.seed)
    if args.model == 'tobit':
        fitted = fit_tobit_gp(prior, data['values'], coordinates=data['coordinates'],
            is_censored=data['is_censored'], threshold=data['threshold'], noise_sd=args.noise_sd,
            measurement_scale=args.measurement_scale, settings=settings, baseline=args.baseline)
    elif args.model == 'poisson':
        fitted = fit_poisson_gp(prior, data['counts'], coordinates=data['coordinates'],
            exposure=data.get('exposure', 1.0), exposure_semantics=args.exposure_semantics,
            settings=settings, baseline=args.baseline)
    else:
        from needlepoint.bounds import SpatialBounds
        bounds = (SpatialBounds.from_xxyy(data['bounds']) if args.bounds_order == 'xxyy'
                  else tuple(data['bounds']))
        events = PointPattern(data['events'], bounds,
            effort=data.get('event_effort'), observed_area=data.get('observed_area'))
        quadrature = Quadrature(data['quadrature_coordinates'], data['quadrature_areas'],
            bounds, effort=data.get('quadrature_effort', 1.0),
            grid_shape=None if 'quadrature_grid_shape' not in data else tuple(data['quadrature_grid_shape']),
            acquired_mask=data.get('acquired_mask'))
        fitted = fit_lgcp(prior, events, quadrature, settings=settings, baseline=args.baseline)
    from needlepoint.storage import VIStorageSettings
    storage = VIStorageSettings(**{k:getattr(args, k) for k in VIStorageSettings.__dataclass_fields__})
    output = fitted.save(args.output, storage=storage)
    print(json.dumps(dict(output=str(output), posterior_identity=fitted.posterior_identity,
                          diagnostics=fitted.diagnostics), indent=2))


if __name__ == '__main__':
    main()
