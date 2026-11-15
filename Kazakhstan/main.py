"""Experimental forensic localizer; output is not a verified flag solution.

Run: python main.py --data /path/to/data --output result
Requires the tesseract binary and its English language data.
Uses RGB residual/ELA statistics and generic Noiseprint++ features, supervised
by the supplied exhibit masks. A region ranker uses out-of-fold pixel maps.
Internal OCR glyph similarity adds proposals for duplicated nonzero digits.
No reference stamps or catalog images are used. boxes.csv contains page pixels.
The development region-center score was 17/30; overlays remain ambiguous.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import argparse
import csv
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import cv2
import joblib
import numpy as np
import torch
from detector import train, detect
from ranker import train_ranker, proposals
import glyphs


def localize(fields, model):
    rows = []
    model.n_jobs = 1
    for path in sorted(fields.glob('*/*.npz')):
        z = np.load(path)
        pb, pn = z['base'], z['noise']
        points = []
        for bx, by, bw, bh in z['boxes']:
            x0, y0 = max(0, (bx-36)//3), max(0, (by-36)//3)
            x1, y1 = min(pb.shape[1], (bx+bw+36)//3), min(pb.shape[0], (by+bh+36)//3)
            b, n = pb[y0:y1, x0:x1], pn[y0:y1, x0:x1]
            valid = np.zeros(b.shape, bool)
            valid[max(0, (by+15)//3-y0):min(b.shape[0], (by+bh-15)//3-y0),
                  max(0, (bx+15)//3-x0):min(b.shape[1], (bx+bw-15)//3-x0)] = True
            box = bx-x0*3, by-y0*3, bw, bh
            for cx, cy, f in proposals(b, n, valid, box):
                points.append((int((cx+x0)*3), int((cy+y0)*3), f))
        scores = model.predict_proba(np.stack([p[2] for p in points]))[:, 1]
        best = int(np.argmax(scores))
        cx, cy, _ = points[best]
        rows.append(dict(album=path.parent.name, page=path.stem, x=cx-55,
            y=cy-45, w=110, h=90, score=float(scores[best]), method='forensic_ranker'))
    return rows


def apply_glyphs(rows, data, cache):
    glyphs.ROOT = cache
    cache.mkdir(parents=True, exist_ok=True)
    paths = sorted((data/'albums').glob('*/*.jpg'))
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(glyphs.process, paths))
    for r in rows:
        im = cv2.imread(str(data/'albums'/r['album']/(r['page']+'.jpg')))
        ps = [p for p in glyphs.clone_proposals(im, r['album']+'_'+r['page'], cache)
              if p[0] >= .9 and p[1] >= 25]
        if ps:
            sim, conf, (x, y, w, h), text, i, j = ps[0]
            r.update(x=int(x-8), y=int(y-8), w=int(w+16), h=int(h+16),
                     score=sim, method='glyph_clone')
    return rows


def save(rows, output):
    if len(rows) != 130 or len({(r['album'], r['page']) for r in rows}) != 130:
        raise ValueError('Expected one result for each of 130 distinct pages')
    with (output/'boxes.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['album','page','x','y','w','h','score','method'])
        writer.writeheader()
        writer.writerows(rows)
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="290" viewBox="0 0 1600 290">',
           '<rect width="1600" height="290" fill="white"/>']
    for r in rows:
        panel = int(r['album'].split('_')[-1])-1
        cx = panel*320+(r['x']+r['w']/2)/5
        cy = 30+(r['y']+r['h']/2)/5
        color = '#007f70' if r['method']=='glyph_clone' else '#222'
        svg.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="3" fill="{color}"/>')
    for i in range(5):
        svg.append(f'<text x="{i*320+10}" y="20" font-size="15">album_{i+1}</text>')
    svg.append('</svg>')
    (output/'overlay.svg').write_text('\n'.join(svg))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('result'))
    args = parser.parse_args()
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    cache = args.output/'cache'
    cache.mkdir(exist_ok=True)
    fields = args.output/'fields'
    train(args.data, cache)
    ranker = train_ranker(args.data, cache)
    detect(args.data, cache, fields)
    rows = apply_glyphs(localize(fields, ranker), args.data, cache/'glyphs')
    save(rows, args.output)


if __name__ == '__main__':
    main()
