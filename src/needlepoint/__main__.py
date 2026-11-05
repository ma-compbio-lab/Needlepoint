"""Unified command dispatcher; importing the package never initializes CUDA."""
import argparse
from importlib import import_module
import sys

COMMANDS = {
    'run': 'cli.run_workflow',
    'score': 'scoring.multiscale_contrast',
    'screen': 'cli.run_direct_ski_screen',
    'fit': 'cli.run_h5ad_joint_global_local',
    'fit-non-gaussian': 'cli.fit_non_gaussian',
    'prototypes': 'cli.run_spatial_prototypes',
    'merge-gp-shards': 'cli.merge_gp_shards',
    'lbm': 'cli.run_lbm',
    'provenance': 'cli.write_provenance',
    'merge-screens': 'cli.merge_lengthscale_screens',
}

def main(argv=None):
    from . import __version__
    parser = argparse.ArgumentParser(prog='needlepoint', description='Needlepoint spatial molecular-field analysis')
    parser.add_argument('--version', action='version', version=__version__)
    parser.add_argument('command', choices=[*COMMANDS, 'defaults'], nargs='?')
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in COMMANDS:
        old_argv = sys.argv
        try:
            sys.argv = ['needlepoint '+args[0], *args[1:]]
            return import_module('needlepoint.'+COMMANDS[args[0]]).main()
        finally:
            sys.argv = old_argv
    parsed = parser.parse_args(args)
    if parsed.command == 'defaults':
        from importlib.resources import files
        print(files('needlepoint').joinpath('resources/default_workflow.toml').read_text(), end='')
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
