"""Point-supervised density forest and image-level count calibration."""
from pathlib import Path
import csv
import cv2
import joblib
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.ensemble import ExtraTreesRegressor
SIZE = 150

def features(im):
 im=cv2.resize(im,(SIZE,SIZE),interpolation=cv2.INTER_AREA).astype(np.float32)/255
 lab=cv2.cvtColor(im,cv2.COLOR_BGR2Lab);lab[:,:,0]/=100;lab[:,:,1:]/=128
 hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV);hsv[:,:,0]/=360
 gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
 out=[im,lab,hsv]
 for sig in [0.6,1.2,2.4,4.8,9.6]:
  smooth=cv2.GaussianBlur(lab,(0,0),sig)
  square=cv2.GaussianBlur(lab*lab,(0,0),sig)
  std=np.sqrt(np.maximum(0,square-smooth*smooth))
  dx=cv2.Sobel(smooth[:,:,0],cv2.CV_32F,1,0);dy=cv2.Sobel(smooth[:,:,0],cv2.CV_32F,0,1)
  mag=cv2.magnitude(dx,dy)
  lap=cv2.Laplacian(smooth[:,:,0],cv2.CV_32F)
  out += [smooth,std,mag[:,:,None],lap[:,:,None],(lab-smooth)]
 return np.concatenate(out,axis=2).reshape(SIZE*SIZE,-1).astype(np.float32)

def new_density_model():
    return ExtraTreesRegressor(n_estimators=48, max_depth=24,
        min_samples_leaf=8, max_features=.8, n_jobs=2, random_state=178)


def aggregate(f, model):
    d = model.predict(f).reshape(SIZE, SIZE)
    extra = []
    for t in [.005, .01, .02, .04, .08, .12]:
        mask = (d > t).astype(np.uint8)
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        areas = stats[1:, cv2.CC_STAT_AREA]
        extra.extend([n-1, mask.sum(), sum(areas > 3), sum(areas > 10), sum(areas > 40)])
    return np.concatenate([f.mean(0), f.std(0),
        np.quantile(f, [.1, .5, .9, .99], axis=0).ravel(),
        [d.sum(), d.std(), d.max()], extra])


def train_models(data, cache):
    cache.mkdir(parents=True, exist_ok=True)
    model_paths = [cache / n for n in ['density_224.joblib', 'density_280.joblib', 'calibrator.joblib']]
    if all(p.exists() for p in model_paths):
        return tuple(joblib.load(p) for p in model_paths)
    paths = sorted((data / 'train').glob('*.jpg'))
    if len(paths) != 280:
        raise ValueError(f'Expected 280 training photographs, found {len(paths)}')
    with open(data / 'train_counts.csv', newline='') as f:
        truth = {r['image']: int(r['count']) for r in csv.DictReader(f)}
    points = {}
    with open(data / 'train_points.csv', newline='') as f:
        for r in csv.DictReader(f):
            points.setdefault(r['image'], []).append((float(r['x']), float(r['y'])))
    rng = np.random.default_rng(178)
    xs, ys, groups = [], [], []
    feature_dir = cache / 'train_features'
    feature_dir.mkdir(exist_ok=True)
    for i, path in enumerate(paths):
        im = cv2.imread(str(path))
        f = features(im)
        np.save(feature_dir / (path.stem + '.npy'), f.astype(np.float16))
        target = np.zeros((SIZE, SIZE), np.float32)
        for x, y in points.get(path.name, []):
            ix = min(SIZE-1, max(0, int(x / im.shape[1] * SIZE)))
            iy = min(SIZE-1, max(0, int(y / im.shape[0] * SIZE)))
            target[iy, ix] += 1
        target = gaussian_filter(target, 1.2, mode='reflect').ravel()
        samples = rng.choice(SIZE * SIZE, 1200, replace=False)
        xs.append(f[samples]); ys.append(target[samples]); groups.extend([i] * len(samples))
    x, y = np.concatenate(xs), np.concatenate(ys)
    valid = np.random.default_rng(178).choice(len(paths), 56, replace=False)
    mask = ~np.isin(groups, valid)
    partial = new_density_model().fit(x[mask], y[mask])
    complete = new_density_model().fit(x, y)
    rows = []
    for path in paths:
        f = np.load(feature_dir / (path.stem + '.npy')).astype(np.float32)
        rows.append(aggregate(f, partial))
    z = np.stack(rows)
    counts = np.array([truth[p.name] for p in paths])
    calibrator = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2,
        max_features=.8, random_state=178, n_jobs=2).fit(z, counts)
    models = (partial, complete, calibrator)
    for model, path in zip(models, model_paths):
        joblib.dump(model, path, compress=3)
    return models
