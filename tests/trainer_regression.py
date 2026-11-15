#!/usr/bin/env python3
"""Check GPU-trained trees using independent CPU split search and traversal.

Uses tiny generated data, no downloads. Verifies every occupied split, heap
posterior, empty-leaf fallback, serialized predictions and odd sample sizes.
"""
import argparse
import collections
import math
import os
from pathlib import Path
import struct
import subprocess
import tempfile


def counts(indices, labels, classes):
    result = [0] * classes
    for i in indices:
        result[labels[i]] += 1
    return result


def entropy(hist):
    total = sum(hist)
    return -sum((x / total) * math.log2(x / total) for x in hist if x) if total else 0.0


def split_cost(indices, feature, threshold, labels, classes, weights):
    left = [i for i in indices if feature[i] <= threshold]
    right = [i for i in indices if feature[i] > threshold]
    left_hist = [n/w if w else 0 for n, w in zip(counts(left, labels, classes), weights)]
    right_hist = [n/w if w else 0 for n, w in zip(counts(right, labels, classes), weights)]
    total = sum(left_hist) + sum(right_hist)
    cost = (sum(left_hist) * entropy(left_hist) + sum(right_hist) * entropy(right_hist)) / total if total else 0.0
    return cost


def verify(directory, features, labels, depth, weight_type, file_weights):
    data = (directory / 'Tree_0000.tree').read_bytes()
    d, classes, node_count, hist_count = struct.unpack_from('<4i', data)
    assert (d, node_count, hist_count) == (depth, 2 ** (depth + 1) - 1, 2 ** (depth + 2) - 1)
    assert classes == max(labels) + 1
    for checkpoint_depth in range(depth + 1):
        path = directory / f'Tree_0000.depth{checkpoint_depth:02d}.tree'
        assert struct.unpack('<i', path.read_bytes()[:4])[0] == checkpoint_depth
    assert (directory / f'Tree_0000.depth{depth:02d}.tree').read_bytes() == data
    assert not list(directory.glob('*.tmp')), 'unfinished atomic snapshot'
    assert len(data) == 16 + 8 * node_count + (8 + classes) * hist_count
    nodes = [struct.unpack_from('<if', data, 16 + 8*i) for i in range(node_count)]
    leaf_offset = 16 + node_count*8
    byte_offset = leaf_offset + hist_count*8
    memberships = [[] for _ in range(hist_count)]
    leaves = []
    for sample in range(len(labels)):
        node = 0
        memberships[0].append(sample)
        while node < node_count:
            f, threshold = nodes[node]
            assert 0 <= f < len(features), (node, f)
            node = 2*node + (2 if features[f][sample] > threshold else 1)
            memberships[node].append(sample)
        leaves.append(node)
    weights = [1] * classes
    if weight_type == 'W_APRIORI':
        weights = counts(range(len(labels)), labels, classes)
    elif weight_type == 'W_FILE':
        weights = file_weights
    expected_hists = []
    for node in range(hist_count):
        hist = counts(memberships[node], labels, classes)
        if not sum(hist) and node:
            hist = expected_hists[(node-1)//2].copy()
        expected_hists.append(hist)
        raw_max, label = struct.unpack_from('<IH', data, leaf_offset + node*8)
        assert raw_max == max(hist), ('max', node, raw_max, max(hist))
        assert label == hist.index(max(hist)), ('label', node, label, hist)
        encoded = list(data[byte_offset + node*classes:byte_offset + (node+1)*classes])
        assert sum(encoded), ('empty posterior', node)
        expected = [int(x*255/max(hist)) for x in hist]
        assert max(abs(a-b) for a, b in zip(encoded, expected)) <= 1, ('histogram', node, encoded, expected)
        if node < node_count and memberships[node]:
            f, threshold = nodes[node]
            cost = split_cost(memberships[node], features[f], threshold, labels, classes, weights)
            # Dataset features are +/-1. Both advertised candidate thresholds
            # (1-epsilon and 0-epsilon) induce the same dichotomy.
            optimal = min(split_cost(memberships[node], feature, 0.0, labels, classes, weights) for feature in features)
            assert cost <= optimal + 2e-5, ('suboptimal split', node, cost, optimal)
    predicted = [expected_hists[node].index(max(expected_hists[node])) for node in leaves]
    saved_labels = list(struct.unpack('<' + 'H'*len(labels), (directory / 'Tree_0000.tree.labels').read_bytes()))
    assert predicted == saved_labels, 'training trajectories differ from serialized inference'
    assert predicted == labels, 'separable generated data must be learned exactly'


def run_case(tea, name, features, labels, depth, folding, pool=None, weight_type="W_ONES", file_weights=None, cache_guard=False):
    with tempfile.TemporaryDirectory(prefix='tea-trainer-') as tmp:
        directory = Path(tmp)
        (directory / 'Labels.lbl').write_bytes(struct.pack('<'+'H'*len(labels), *labels))
        thresholds = []
        for i, feature in enumerate(features):
            (directory / f'F_{i:04d}.feat').write_bytes(struct.pack('<'+'h'*len(feature), *feature))
            thresholds.extend([4., -1., 1.])
        (directory / 'Threshholds.thr').write_bytes(struct.pack('<'+'f'*len(thresholds), *thresholds))
        if file_weights:
            (directory / 'Weights.txt').write_text(''.join(f'{i} {w}\n' for i, w in enumerate(file_weights)))
        command = [str(tea), str(len(features)), str(pool or len(features)), str(depth), str(folding), 'F_SHORT',
                   str(len(labels)), '2', '0', '0', weight_type, str(directory)+'/', 'regression', '0', 'LOG0', name]
        environment = os.environ.copy()
        cache_marker = directory / 'global-cache-attempt'
        if cache_guard:
            fake_bin = directory / 'bin'
            fake_bin.mkdir()
            fake_tee = fake_bin / 'tee'
            fake_tee.write_text('#!/bin/sh\nprintf attempted > "$TEA_CACHE_TEST_MARKER"\nexit 0\n')
            fake_tee.chmod(0o755)
            environment.update(DROP_CACHE='1', TEA_CACHE_TEST_MARKER=str(cache_marker),
                               PATH=str(fake_bin)+os.pathsep+environment.get('PATH', ''))
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120, env=environment)
        assert not cache_marker.exists(), 'DROP_CACHE triggered a global-cache control command'
        if result.returncode:
            raise AssertionError(f'{name}: trainer exit {result.returncode}\n{result.stdout[-6000:]}')
        verify(directory, features, labels, depth, weight_type, file_weights)
        print('PASS:', name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tea', required=True)
    args = parser.parse_args()
    tea = Path(args.tea).resolve()
    run_case(tea, 'odd-root-majority-right', [[-1]*3+[1]*6], [1]*3+[0]*6, 0, 0)
    codes = [code for code in range(16) for _ in range(code % 3 + 1)]
    features = [[1 if code & (1 << bit) else -1 for code in codes] for bit in range(4)]
    run_case(tea, 'odd-deep-empty-leaves', features, codes, 7, 7)
    run_case(tea, 'folded-deep-empty-batches', features, codes, 7, 2)
    run_case(tea, 'streaming-feature-pool', features, codes, 7, 2, pool=1)
    run_case(tea, 'legacy-drop-cache-ignored', features, codes, 3, 2, pool=1, cache_guard=True)
    run_case(tea, 'inverse-prior-weighted-splits', features, codes, 4, 2, pool=1, weight_type='W_APRIORI')
    weighted_features = [[-1]*3+[1]*6, [-1,1,1,-1,-1,-1,1,1,1]]
    run_case(tea, 'tiny-weighted-mass-pure-vs-noise', weighted_features, [1]*3+[0]*6, 0, 0,
             weight_type='W_FILE', file_weights=[100000000,100000000])
    # Labels above the fast-kernel range force the independent regular path.
    labels = [code*20 for code in codes]
    run_case(tea, 'sparse-labels-regular-histogram', features, labels, 4, 2)
    print('PASS: independent CPU split search, all serialized posteriors and held-out-style traversal')


if __name__ == '__main__':
    main()
