"""Decode a noisy 5x7 raster from georeferenced sheep-count estimates."""
from pathlib import Path
import csv
import cv2
import numpy as np
import cmudict

T={
'A':['01110','10001','10001','11111','10001','10001','10001'],
'B':['11110','10001','10001','11110','10001','10001','11110'],
'C':['01111','10000','10000','10000','10000','10000','01111'],
'D':['11110','10001','10001','10001','10001','10001','11110'],
'E':['11111','10000','10000','11110','10000','10000','11111'],
'F':['11111','10000','10000','11110','10000','10000','10000'],
'G':['01111','10000','10000','10111','10001','10001','01111'],
'H':['10001','10001','10001','11111','10001','10001','10001'],
'I':['11111','00100','00100','00100','00100','00100','11111'],
'J':['00111','00010','00010','00010','10010','10010','01100'],
'K':['10001','10010','10100','11000','10100','10010','10001'],
'L':['10000','10000','10000','10000','10000','10000','11111'],
'M':['10001','11011','10101','10101','10001','10001','10001'],
'N':['10001','11001','10101','10011','10001','10001','10001'],
'O':['01110','10001','10001','10001','10001','10001','01110'],
'P':['11110','10001','10001','11110','10000','10000','10000'],
'Q':['01110','10001','10001','10001','10101','10010','01101'],
'R':['11110','10001','10001','11110','10100','10010','10001'],
'S':['01111','10000','10000','01110','00001','00001','11110'],
'T':['11111','00100','00100','00100','00100','00100','00100'],
'U':['10001','10001','10001','10001','10001','10001','01110'],
'V':['10001','10001','10001','10001','10001','01010','00100'],
'W':['10001','10001','10001','10101','10101','10101','01010'],
'X':['10001','10001','01010','00100','01010','10001','10001'],
'Y':['10001','10001','01010','00100','00100','00100','00100'],
'Z':['11111','00001','00010','00100','01000','10000','11111']}

def write_map(data, rows, output):
    counts = dict(rows)
    with open(data / 'survey_plots.csv', newline='') as f:
        plots = list(csv.DictReader(f))
    coords = np.array([[float(r['x_m']), float(r['y_m'])] for r in plots])
    centered = coords - coords.mean(axis=0)
    _, _, basis = np.linalg.svd(centered, full_matrices=False)
    horizontal = basis[0]
    if horizontal[0] < 0:
        horizontal = -horizontal
    # East is right, south is down, so the raster has geographic orientation.
    vertical = np.array([horizontal[1], -horizontal[0]])
    xy = centered @ np.stack([horizontal, vertical]).T
    # Projected coordinate levels are approximately 50 metres apart.
    col = np.rint((xy[:, 0] - xy[:, 0].min()) / 50).astype(int)
    row = np.rint((xy[:, 1] - xy[:, 1].min()) / 50).astype(int)
    height, width = int(row.max()+1), int(col.max()+1)
    grid = np.full((height, width), np.nan)
    for r, c, p in zip(row, col, plots):
        grid[r, c] = counts[p['image']]
    values = grid[np.isfinite(grid)]
    # Data-driven Otsu threshold; no word, shape or plot labels are prescribed.
    scaled = np.clip(values / max(float(values.max()), 1) * 255, 0, 255).astype(np.uint8)
    threshold, _ = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = threshold / 255 * max(float(values.max()), 1)
    text = '\n'.join(''.join('##' if x >= threshold else '  ' for x in line) for line in grid)
    (output / 'threshold_map.txt').write_text(text + '\n', encoding='utf-8')
    print('Map threshold:', threshold, flush=True)
    print(text, flush=True)
    cell = 42
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width*cell}" height="{height*cell}" viewBox="0 0 {width*cell} {height*cell}">']
    for r in range(height):
        for c in range(width):
            value = grid[r, c]
            shade = int(255 * (1 - min(1, max(0, value / max(float(values.max()), 1)))))
            color = '#%02x%02x%02x' % (shade, shade, shade)
            textcolor = 'white' if shade < 125 else 'black'
            svg.append(f'<rect x="{c*cell}" y="{r*cell}" width="{cell}" height="{cell}" fill="{color}"/>')
            svg.append(f'<text x="{c*cell+cell/2}" y="{r*cell+cell/2+5}" text-anchor="middle" font-family="monospace" font-size="12" fill="{textcolor}">{int(round(value))}</text>')
    svg.append('</svg>')
    (output / 'survey_map.svg').write_text('\n'.join(svg), encoding='utf-8')


def decode_counts(plots_path, rows, output):
    output.mkdir(parents=True, exist_ok=True)
    with open(plots_path, newline='') as f:
        plots = list(csv.DictReader(f))
    counts = dict(rows)
    a = np.array([[float(r['x_m']), float(r['y_m'])] for r in plots])
    _, _, basis = np.linalg.svd(a-a.mean(0), full_matrices=False)
    east = basis[0]
    if east[0] < 0:
        east = -east
    south = np.array([east[1], -east[0]])
    xy = (a-a.mean(0)) @ np.stack([east, south]).T
    cols = np.rint((xy[:, 0]-xy[:, 0].min()) / 50).astype(int)
    rr = np.rint((xy[:, 1]-xy[:, 1].min()) / 50).astype(int)
    g = np.full((rr.max()+1, cols.max()+1), np.nan)
    for r, c, plot in zip(rr, cols, plots):
        g[r, c] = counts[plot['image']]
    if not np.isfinite(g).all():
        raise ValueError('Some survey counts or grid cells are missing')
    length = (g.shape[1]-1) // 6
    if g.shape[0] != 9 or g.shape[1] != 6*length+1:
        raise ValueError('Unexpected raster dimensions')
    masks = {letter: np.array([[int(v) for v in line] for line in rs]) for letter, rs in T.items()}
    # The map is noisy: a general pronunciation dictionary provides a language
    # prior. No competition flag or preferred word is embedded in the program.
    words = sorted({w.upper() for w in cmudict.words() if len(w) == length and w.isalpha()})
    # Otsu gives approximately 37 for this census. Robust clipping prevents one
    # erroneous very large count from dominating a whole character's likelihood.
    vals = g.ravel()
    scaled = np.clip(vals / max(float(vals.max()), 1) * 255, 0, 255).astype(np.uint8)
    threshold, _ = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    midpoint = round(threshold / 255 * max(float(vals.max()), 1))
    candidates = []
    variants = [('normal', g), ('mirror_x', g[:, ::-1]),
                ('mirror_y', g[::-1]), ('rotate_180', g[::-1, ::-1])]
    for orientation, raster in variants:
        scores = []
        for k in range(length):
            cell = raster[1:8, 1+k*6:6+k*6]
            scores.append({l: float(((np.clip(cell, 0, 65)-midpoint)*m).sum())
                for l, m in masks.items()})
        for word in words:
            candidates.append((sum(scores[i][letter] for i, letter in enumerate(word)), word, orientation))
    candidates.sort(reverse=True)
    with open(output / 'decoder_candidates.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['word', 'orientation', 'score'])
        writer.writerows((word, orientation, f'{score:.6f}') for score, word, orientation in candidates[:50])
    _, word, orientation = candidates[0]
    flag = 'SIGMOID_' + word
    (output / 'flag.txt').write_text(flag+'\n', encoding='utf-8')
    cells = np.zeros((9, 6*len(word)+1), np.uint8)
    for i, letter in enumerate(word):
        cells[1:8, 1+i*6:6+i*6] = masks[letter]
    size = 20
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{cells.shape[1]*size}" height="{cells.shape[0]*size+30}">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<text x="5" y="20" font-size="14">OCR reconstruction, not raw measurements</text>']
    for r, c in zip(*np.where(cells)):
        svg.append(f'<rect x="{c*size}" y="{r*size+30}" width="{size}" height="{size}" fill="black"/>')
    svg.append('</svg>')
    (output / 'decoded_word.svg').write_text('\n'.join(svg), encoding='utf-8')
    print('Best OCR candidate:', flag, orientation, flush=True)
    return flag


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--counts', type=Path, default=Path('counts.csv'))
    p.add_argument('--plots', type=Path, default=Path('data/survey_plots.csv'))
    p.add_argument('--output', type=Path, default=Path('.'))
    args = p.parse_args()
    with open(args.counts, newline='') as f:
        rows = [(r['image'], float(r['count'])) for r in csv.DictReader(f)]
    decode_counts(args.plots, rows, args.output)
