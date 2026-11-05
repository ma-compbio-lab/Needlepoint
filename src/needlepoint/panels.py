"""Exact ordered gene-panel validation shared by dataset adapters."""
import os
from pathlib import Path
import shutil

from needlepoint.artifacts import sha256_file


def audit_fixed_panel(path, input_h5ad=None, *, expected_sha256=None):
    import pandas as pd
    path = Path(path)
    digest = sha256_file(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError('Fixed gene panel checksum changed')
    table = pd.read_csv(path, dtype={'gene': str}, keep_default_na=False)
    if 'gene' not in table or table.empty:
        raise ValueError('Fixed gene panel must contain a nonempty gene column')
    genes = table['gene'].tolist()
    if any(not gene or gene != gene.strip() for gene in genes) or len(set(genes)) != len(genes):
        raise ValueError('Fixed gene panel contains blank, padded, or duplicate gene identifiers')
    result = dict(path=str(path.resolve()), sha256=digest, n_genes=len(genes),
                  policy='exact-reference-panel-no-intersection-no-reselection')
    if input_h5ad is not None:
        import h5py
        try:
            from anndata.io import read_elem
        except ImportError:  # Public location in AnnData 0.10.
            from anndata.experimental import read_elem
        with h5py.File(input_h5ad, 'r') as h:
            # AnnData indices may be string datasets or encoded groups
            # (e.g. nullable strings with newer Pandas). Use its decoder.
            names = read_elem(h['var'][h['var'].attrs['_index']]).tolist()
        if len(set(names)) != len(names):
            raise ValueError('Input gene identifiers must be unique')
        available = set(names)
        missing = [gene for gene in genes if gene not in available]
        if missing:
            raise ValueError(f'Input lacks {len(missing)} fixed-panel genes; refusing to drop or substitute them: {missing[:20]}')
        result.update(input_h5ad=str(Path(input_h5ad).resolve()), all_genes_present=True)
    return result


def materialize_fixed_panel(source, destination, input_h5ad, *, expected_sha256):
    """Copy exact panel bytes atomically; an existing differing panel is refused."""
    audit = audit_fixed_panel(source, input_h5ad, expected_sha256=expected_sha256)
    destination = Path(destination)
    if destination.exists():
        if sha256_file(destination) != expected_sha256:
            raise ValueError('Existing run gene panel differs from the frozen reference')
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f'.{destination.name}.partial-{os.getpid()}')
        try:
            shutil.copy2(source, temporary)
            if sha256_file(temporary) != expected_sha256:
                raise ValueError('Reference panel changed during copy')
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
    return dict(audit, output_panel=str(destination.resolve()))
