import csv
from pathlib import Path
import cv2
import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from features import stamps_on, rgb_features, features_from_noise
from noiseprint import extract_noise, load_noiseprint


def new_classifier(trees=96):
    return ExtraTreesClassifier(n_estimators=trees, max_depth=24,
        min_samples_leaf=5, max_features=.8, n_jobs=2,
        random_state=178, class_weight='balanced')


def train(data, cache):
    files = sorted((data / 'exhibits').glob('*.jpg'))
    if len(files) != 30:
        raise ValueError('Expected 30 labelled exhibits')
    model_paths = [cache / 'rgb.joblib', cache / 'noise.joblib']
    if all(p.exists() for p in model_paths):
        return tuple(joblib.load(p) for p in model_paths)
    npp = load_noiseprint(cache, fused=False)
    xs, ys = [], []
    rng = np.random.default_rng(178)
    for path in files:
        im = cv2.imread(str(path))
        mask = cv2.imread(str(path.with_name(path.stem + '_mask.png')), 0)
        if im is None or mask is None:
            raise ValueError('Missing image/mask: ' + path.name)
        target = mask > 0
        stored = cache / ('training_noise_' + path.stem + '.npy')
        if stored.exists():
            noise = np.load(stored)
        else:
            noise = extract_noise(npp, im)
            np.save(stored, noise)
        f = features_from_noise(im, noise)
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
    x, y = np.concatenate(xs), np.concatenate(ys)
    rgb = new_classifier().fit(x[:, :85], y)
    noise = new_classifier().fit(x, y)
    for model, path in zip([rgb, noise], model_paths):
        joblib.dump(model, path, compress=3)
    return rgb, noise


def detect(data, cache, field_dir):
    paths = sorted((data / 'albums').glob('*/*.jpg'))
    if len(paths) != 130:
        raise ValueError(f'Expected 130 pages, found {len(paths)}')
    if all((field_dir / p.parent.name / (p.stem + '.npz')).exists() for p in paths):
        return
    rgb_model, noise_model = train(data, cache)
    rgb_model.n_jobs = noise_model.n_jobs = 1
    npp = load_noiseprint(cache, fused=True)
    for i, path in enumerate(paths):
        target = field_dir / path.parent.name / (path.stem + '.npz')
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            continue
        im = cv2.imread(str(path))
        boxes = stamps_on(im)
        page = np.zeros(im.shape[:2], np.float32)
        base = np.zeros_like(page)
        expert = np.zeros_like(page)
        for j, (bx, by, bw, bh) in enumerate(boxes):
            # Preserve the phase of the page's JPEG block grid.
            x0, y0 = max(0, (bx-24)//8*8), max(0, (by-24)//8*8)
            x1 = min(im.shape[1], (bx+bw+31)//8*8)
            y1 = min(im.shape[0], (by+bh+31)//8*8)
            crop = im[y0:y1, x0:x1]
            saved = cache / (path.parent.name + '_' + path.stem + '_' + str(j) + '.npy')
            if saved.exists():
                noise = np.load(saved)
            else:
                noise = extract_noise(npp, crop)
                np.save(saved, noise)
            f = features_from_noise(crop, noise)
            b = rgb_model.predict_proba(f[:, :, :85].reshape(-1, 85))[:, 1].reshape(f.shape[:2])
            n = noise_model.predict_proba(f.reshape(-1, f.shape[-1]))[:, 1].reshape(f.shape[:2])
            p = cv2.resize(np.sqrt(b*n), (x1-x0, y1-y0))
            page[by+15:by+bh-15, bx+15:bx+bw-15] = p[
                by+15-y0:by+bh-15-y0, bx+15-x0:bx+bw-15-x0]
            for field, raw in [(base, b), (expert, n)]:
                raw = cv2.resize(raw, (x1-x0, y1-y0))
                field[by+15:by+bh-15, bx+15:bx+bw-15] = raw[by+15-y0:by+bh-15-y0, bx+15-x0:bx+bw-15-x0]
        np.savez_compressed(target, prob=page[::3, ::3], base=base[::3, ::3], noise=expert[::3, ::3], boxes=np.array(boxes))
        print(f'Detected page {i+1}/130', flush=True)
