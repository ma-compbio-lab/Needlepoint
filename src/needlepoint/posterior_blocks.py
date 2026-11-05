"""Read canonical NPZ moments without an expanded on-disk cache.

New exports use ZIP_STORED NPY members, which can be mapped directly. Legacy
compressed exports remain readable, with at most one decompressed chunk cached
in RAM. No extraction or persistent copy is performed.
"""
import os
import struct
import zipfile
from pathlib import Path

import numpy as np


def atomic_savez_maps(path, **arrays):
    path = Path(path)
    temporary = path.with_name(f'.{path.stem}.partial-{os.getpid()}.npz')
    try:
        np.savez(temporary, **arrays)
        with temporary.open('rb') as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class PosteriorBlocks:
    def __init__(self):
        self._mapped = {}
        self._legacy_path = None
        self._legacy = {}

    def array(self, path, name):
        path = Path(path)
        key = (path, name)
        if key in self._mapped:
            return self._mapped[key]
        if self._legacy_path == path and name in self._legacy:
            return self._legacy[name]
        with zipfile.ZipFile(path) as archive:
            info = archive.getinfo(name + '.npy')
            if info.compress_type != zipfile.ZIP_STORED:
                if self._legacy_path != path:
                    self._legacy.clear()
                    self._legacy_path = path
                with archive.open(info) as handle:
                    array = np.lib.format.read_array(handle, allow_pickle=False)
                self._validate(array.shape, array.dtype)
                self._legacy[name] = array
                return array
            with path.open('rb') as handle:
                handle.seek(info.header_offset)
                header = handle.read(30)
                if len(header) != 30 or header[:4] != b'PK\x03\x04':
                    raise ValueError('Invalid NPZ local header')
                name_size, extra_size = struct.unpack_from('<HH', header, 26)
                start = info.header_offset + 30 + name_size + extra_size
                handle.seek(start)
                version = np.lib.format.read_magic(handle)
                if version == (1, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
                elif version == (2, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle)
                else:
                    raise ValueError(f'Unsupported numeric NPY version: {version}')
                self._validate(shape, dtype)
                offset = handle.tell()
                if offset + int(np.prod(shape)) * dtype.itemsize != start + info.file_size:
                    raise ValueError('NPZ member size does not match its array header')
        array = np.memmap(path, mode='r', dtype=dtype, shape=shape,
                          offset=offset, order='F' if fortran else 'C')
        self._mapped[key] = array
        return array

    @staticmethod
    def _validate(shape, dtype):
        if len(shape) != 2 or dtype.kind != 'f':
            raise ValueError('Posterior maps must be two-dimensional floating arrays')

    def read(self, path, name, rows, columns, dtype=np.float32):
        return np.asarray(self.array(path, name)[np.ix_(rows, columns)], dtype=dtype)
