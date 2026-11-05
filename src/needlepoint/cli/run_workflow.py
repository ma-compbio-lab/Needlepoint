"""Plan or execute the public sequencing-count workflow."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import tomllib


def main():
    from needlepoint import Resources, Workflow, WorkflowConfig
    parser = argparse.ArgumentParser(description='Plan a Needlepoint workflow; --execute starts fitting. Input H5AD is loaded in RAM.')
    parser.add_argument('--h5ad',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--config',type=Path,help='TOML with [workflow] and [resources] tables')
    parser.add_argument('--coord-key',default='auto')
    parser.add_argument('--coordinate-scale',type=float,default=1.,help='Coordinate multiplier converting source units to micrometres')
    parser.add_argument('--layer',default=None)
    parser.add_argument('--device',default=None,help='auto, cpu, cuda, or cuda:<visible ordinal>')
    parser.add_argument('--gene-panel',type=Path,help='CSV with a gene column; explicit fixed-panel mode')
    parser.add_argument('--stop-after',choices=['score','screen','gp','prototypes','lbm'])
    parser.add_argument('--input-storage', choices=['copy', 'reference'])
    parser.add_argument('--retain-caches', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    if args.resume and not args.execute: parser.error('--resume requires --execute')
    options = tomllib.loads(args.config.read_text()) if args.config else {}
    if set(options)-{'workflow','resources'}: parser.error('Config tables must be [workflow] and [resources]')
    config = WorkflowConfig(**options.get('workflow',{}))
    resources = Resources(**options.get('resources',{}))
    if args.device is not None: resources = replace(resources,device=args.device)
    if args.stop_after is not None: config = replace(config,stop_after=args.stop_after)
    if args.input_storage is not None: config = replace(config,input_storage=args.input_storage)
    if args.retain_caches is not None: config = replace(config,retain_caches=args.retain_caches)
    if args.gene_panel:
        import pandas as pd
        config = replace(config,selected_genes=tuple(pd.read_csv(args.gene_panel)['gene'].astype(str)))
    workflow = Workflow(args.h5ad,args.output_dir,config=config,resources=resources,
                        coord_key=args.coord_key,coordinate_scale=args.coordinate_scale,layer=args.layer)
    if args.execute:
        result = workflow.run(resume=args.resume)
        print(json.dumps({'output_dir':str(result.output_dir),'status':result.state['status']},indent=2))
    else:
        print(json.dumps(workflow.plan(),indent=2))


if __name__ == '__main__':
    main()
