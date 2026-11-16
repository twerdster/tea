#!/usr/bin/env python3
"""Bounded 9-sample test of the exact 20-decision dense allocation path."""
import argparse
import mmap
from pathlib import Path
import struct
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tea', required=True)
    args = parser.parse_args()
    tea = Path(args.tea).resolve()
    with tempfile.TemporaryDirectory(prefix='tea-depth20-') as tmp:
        root = Path(tmp)
        labels = [1]*3 + [0]*6
        feature = [-1]*3 + [1]*6
        (root/'Labels.lbl').write_bytes(struct.pack('<9H', *labels))
        (root/'F_0000.feat').write_bytes(struct.pack('<9h', *feature))
        (root/'Threshholds.thr').write_bytes(struct.pack('<3f', 4., -1., 1.))
        command = [str(tea), '1', '1', '19', '8', 'F_SHORT', '9', '2', '0', '0',
                   'W_ONES', str(root)+'/', 'depth20', '0', 'LOG0', '9-sample allocation/serialization test']
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=240)
        assert result.returncode == 0, result.stdout[-8000:]
        with (root/'Tree_0000.tree').open('rb') as handle:
            with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as model:
                depth, classes, nodes, hists = struct.unpack_from('<4i', model)
                assert (depth, classes, nodes, hists) == (19, 2, 1048575, 2097151)
                leaf_offset, byte_offset = 16+8*nodes, 16+8*nodes+8*hists
                assert len(model) == byte_offset+2*hists
                assert model[byte_offset:byte_offset+2] == bytes([255, 127])
                for i in range(hists):
                    assert model[byte_offset+2*i] or model[byte_offset+2*i+1], ('empty posterior', i)
                predictions = []
                for value in feature:
                    node = 0
                    for _ in range(20):
                        f, threshold = struct.unpack_from('<if', model, 16+8*node)
                        assert f == 0
                        node = 2*node + (2 if value > threshold else 1)
                    _, label = struct.unpack_from('<IH', model, leaf_offset+8*node)
                    predictions.append(label)
                assert predictions == labels
        assert list(struct.unpack('<9H', (root/'Tree_0000.tree.labels').read_bytes())) == labels
        for depth in range(20):
            with (root/f'Tree_0000.depth{depth:02d}.tree').open('rb') as handle:
                assert struct.unpack('<i', handle.read(4))[0] == depth
        assert not list(root.glob('*.tmp'))
        print('PASS: 20 decisions (CLI19), 1,048,575 split nodes, 2,097,151 posteriors, 20 atomic checkpoints, odd samples')


if __name__ == '__main__':
    main()
