from pathlib import Path
import cv2
import numpy as np
import lightgbm as lgb
from booster import forensic_features, model
from features import stamps_on
from noiseprint import extract_noise, load_noiseprint


def train(data, cache, destination):
    paths = sorted((data/'exhibits').glob('*.jpg'))
    if len(paths) != 30:
        raise ValueError('Expected 30 labelled exhibits')
    npp = load_noiseprint(cache, fused=False)
    xs, ys = [], []
    rng = np.random.default_rng(178)
    for path in paths:
        im = cv2.imread(str(path))
        target = cv2.imread(str(path.with_name(path.stem+'_mask.png')), 0) > 0
        noise = extract_noise(npp, im)
        f = forensic_features(im, noise).astype(np.float16).astype(np.float32)
        valid = np.zeros(target.shape, np.uint8)
        for x, y, w, h in stamps_on(im):
            valid[y+15:y+h-15, x+15:x+w-15] = 1
        labels = target[::3, ::3].ravel()
        inside = valid[::3, ::3].ravel() > 0
        pos = np.flatnonzero(labels & inside)
        neg = np.flatnonzero(~labels & inside)
        neg = rng.choice(neg, min(len(neg), 3500), replace=False)
        take = np.r_[pos, neg]
        xs.append(f.reshape(-1, f.shape[-1])[take])
        ys.append(labels[take])
    classifier = model().fit(np.concatenate(xs), np.concatenate(ys))
    classifier.booster_.save_model(str(destination))


def detect(data, cache, fields, weights):
    paths = sorted((data/'albums').glob('*/*.jpg'))
    if len(paths) != 130:
        raise ValueError(f'Expected 130 pages; found {len(paths)}')
    if all((fields/p.parent.name/(p.stem+'.npz')).exists() for p in paths):
        return
    if not weights.exists():
        train(data, cache, weights)
    classifier = lgb.Booster(model_file=str(weights))
    npp = load_noiseprint(cache, fused=True)
    for i, path in enumerate(paths):
        destination = fields/path.parent.name/(path.stem+'.npz')
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            continue
        im = cv2.imread(str(path))
        page = np.zeros(im.shape[:2], np.float32)
        boxes = stamps_on(im)
        for bx, by, bw, bh in boxes:
            # Keep the page JPEG grid phase when cutting out a stamp.
            x0, y0 = max(0, (bx-24)//8*8), max(0, (by-24)//8*8)
            x1 = min(im.shape[1], (bx+bw+31)//8*8)
            y1 = min(im.shape[0], (by+bh+31)//8*8)
            crop = im[y0:y1, x0:x1]
            noise = extract_noise(npp, crop)
            f = forensic_features(crop, noise).astype(np.float16).astype(np.float32)
            p = classifier.predict(f.reshape(-1, f.shape[-1]), num_threads=2)
            p = p.reshape(f.shape[:2]).astype(np.float32)
            p = cv2.resize(p, (x1-x0, y1-y0))
            page[by+15:by+bh-15, bx+15:bx+bw-15] = p[
                by+15-y0:by+bh-15-y0, bx+15-x0:bx+bw-15-x0]
        np.savez_compressed(destination, prob=page[::3, ::3], boxes=np.array(boxes))
        print(f'Page {i+1}/130', flush=True)
