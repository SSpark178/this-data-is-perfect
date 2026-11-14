"""Reassemble vertical image strips using boundary continuity and path assignment."""
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

cv2.setNumThreads(1)


def assemble_path(cost):
    n = len(cost)
    matrix = np.zeros((n + 1, n + 1), dtype=np.float64)
    matrix[:n, :n] = cost
    np.fill_diagonal(matrix, 1e30)
    rows, cols = linear_sum_assignment(matrix)
    successor = np.empty(n + 1, dtype=int)
    successor[rows] = cols
    # Join assignment cycles by the least expensive exchange of outgoing edges.
    while True:
        labels = np.full(n + 1, -1, dtype=int)
        groups = []
        for start in range(n + 1):
            if labels[start] >= 0:
                continue
            group, node = [], start
            while labels[node] < 0:
                labels[node] = len(groups)
                group.append(node)
                node = successor[node]
            groups.append(group)
        if len(groups) == 1:
            break
        nxt = successor
        delta = matrix[:, nxt].copy()
        delta += matrix[:, nxt].T
        old = matrix[np.arange(n + 1), nxt]
        delta -= old[:, None] + old[None, :]
        delta[labels[:, None] == labels[None, :]] = 1e30
        a, b = np.unravel_index(np.argmin(delta), delta.shape)
        successor[a], successor[b] = successor[b], successor[a]
    order = []
    node = successor[n]
    while node != n:
        order.append(int(node))
        node = successor[node]
    return order


def improve(order, cost):
    n = len(order)
    c = np.zeros((n + 1, n + 1))
    c[:n, :n] = cost
    # Relocate contiguous blocks while retaining their internal pixel order.
    for _ in range(8):
        changed = False
        for length in (1, 2, 3, 5, 8, 13, 21):
            if length >= n:
                continue
            for start in range(n - length + 1):
                block = order[start:start + length]
                rest = order[:start] + order[start + length:]
                before = order[start - 1] if start else n
                after = order[start + length] if start + length < n else n
                saving = c[before, block[0]] + c[block[-1], after] - c[before, after]
                left = np.array([n] + rest)
                right = np.array(rest + [n])
                delta = c[left, block[0]] + c[block[-1], right] - c[left, right] - saving
                pos = int(np.argmin(delta))
                if delta[pos] < -1e-6:
                    order = rest[:pos] + block + rest[pos:]
                    changed = True
        if not changed:
            break
    return order


def main():
    source = Path('input')
    files = sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'})
    strips = []
    for p in files:
        im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if im is None:
            raise ValueError('Unreadable image: ' + p.name)
        strips.append(im)
    if not strips:
        raise ValueError('No images found in input/')
    n = len(strips)
    if n == 1:
        order = [0]
    else:
        edges = []
        for im in strips:
            if im.ndim == 2:
                rgb = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
            else:
                rgb = im[:, :, :3]
            rgb = rgb.astype(np.float32)
            if rgb.shape[0] > 1200:
                rgb = cv2.resize(rgb, (rgb.shape[1], 1200), interpolation=cv2.INTER_AREA)
            rgb = cv2.GaussianBlur(rgb, (1, 3), 0.65)
            left, right = rgb[:, 0], rgb[:, -1]
            gl = rgb[:, min(1, rgb.shape[1]-1)] - left
            gr = right - rgb[:, max(0, rgb.shape[1]-2)]
            edges.append((left.ravel(), right.ravel(), gl.ravel(), gr.ravel()))
        l, r, gl, gr = [np.stack([e[k] for e in edges]) for k in range(4)]
        cost = cdist(r, l, 'cityblock')
        cost += 0.5 * cdist(r + gr, l, 'cityblock')
        cost += 0.5 * cdist(r, l - gl, 'cityblock')
        np.fill_diagonal(cost, 1e30)
        order = improve(assemble_path(cost), cost)
    result = np.concatenate([strips[i] for i in order], axis=1)
    if not cv2.imwrite('reconstructed_image.png', result):
        raise RuntimeError('Could not save reconstructed_image.png')
    with open('reconstructed_order.json', 'w', encoding='utf-8') as f:
        json.dump([files[i].name for i in order], f, indent=2)


if __name__ == '__main__':
    main()
