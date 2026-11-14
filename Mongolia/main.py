"""Reproduce the Mongolia sheep census and decode its geographical raster.
Run: python main.py --download
"""
import os
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
import argparse
import csv
from pathlib import Path
import cv2
import numpy as np
from density import train_models, features, aggregate, SIZE
from matching import PointTransfer
from decode import write_map, decode_counts
cv2.setNumThreads(1)
cv2.setRNGSeed(178)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('data'))
    p.add_argument('--cache', type=Path, default=Path('.cache_mongolia'))
    p.add_argument('--output', type=Path, default=Path('.'))
    p.add_argument('--download', action='store_true')
    args = p.parse_args()
    if args.download:
        import gdown
        args.data.mkdir(parents=True, exist_ok=True)
        gdown.download_folder('https://drive.google.com/drive/folders/1acU-wiKH9ar7uTmg3m8-v2aAtEvT4k9e',
            output=str(args.data), quiet=False)
    paths = sorted((args.data / 'survey').glob('*.jpg'))
    if len(paths) != 333:
        raise ValueError(f'Expected 333 survey images, found {len(paths)}')
    print('Preparing density and calibration models...', flush=True)
    partial, complete, calibrator = train_models(args.data, args.cache)
    print('Preparing photograph retrieval...', flush=True)
    transfer = PointTransfer(args.data, args.cache)
    rows, details = [], []
    for i, path in enumerate(paths):
        im = cv2.imread(str(path))
        if im is None:
            raise ValueError('Cannot read '+str(path))
        f = features(im)
        calibrated = max(0.0, float(calibrator.predict(aggregate(f, partial)[None])[0]))
        match = transfer.find(i)
        donor = ''; coverage = correlation = inliers = ''
        if match:
            _, donor, number, mask, coverage, correlation, inliers = match
            density = complete.predict(f).reshape(SIZE, SIZE)
            known = cv2.resize(mask.astype(np.float32), (SIZE, SIZE), interpolation=cv2.INTER_AREA)
            count = number + float((density*(1-known)).sum())
            method = 'point_transfer'
        else:
            count = calibrated
            method = 'calibrated_density'
        rows.append((path.name, max(0.0, count)))
        details.append((path.name, f'{count:.6f}', method, donor, coverage, correlation, inliers))
        if i % 25 == 0:
            print(f'Counted {i+1}/333 photographs', flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    with open(args.output / 'counts.csv', 'w', newline='') as f:
        writer = csv.writer(f); writer.writerow(['image', 'count'])
        writer.writerows((name, int(round(count))) for name, count in rows)
    with open(args.output / 'counts_details.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image', 'estimated_count', 'method', 'donor', 'coverage', 'correlation', 'inliers'])
        writer.writerows(details)
    write_map(args.data, rows, args.output)
    decode_counts(args.data / 'survey_plots.csv', rows, args.output)


if __name__ == '__main__':
    main()
