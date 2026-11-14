"""Retrieve overlapping photographs and transfer annotated sheep centres."""
import csv
import cv2
import joblib
import numpy as np
from sklearn.cluster import MiniBatchKMeans


def root_sift(im, sift):
    keypoints, d = sift.detectAndCompute(im, None)
    if d is None:
        return np.empty((0, 2), np.float32), np.empty((0, 128), np.float32)
    p = np.float32([k.pt for k in keypoints])
    d = np.sqrt(d / (d.sum(1, keepdims=True) + 1e-8)).astype(np.float32)
    return p, d


def matcher(descriptors, checks=40):
    m = cv2.FlannBasedMatcher(dict(algorithm=1, trees=3), dict(checks=checks))
    m.add([descriptors]); m.train()
    return m


class PointTransfer:
    def __init__(self, data, cache):
        self.train = sorted((data / 'train').glob('*.jpg'))
        self.survey = sorted((data / 'survey').glob('*.jpg'))
        self.paths = self.train + self.survey
        self.sift = cv2.SIFT_create(nfeatures=1200, contrastThreshold=.025)
        saved = cache / 'sift.joblib'
        if saved.exists():
            self.fs = joblib.load(saved)
        else:
            self.fs = [root_sift(cv2.imread(str(p), 0), self.sift) for p in self.paths]
            joblib.dump(self.fs, saved)
        self.n = len(self.train)
        self.local = [matcher(d) for _, d in self.fs[:self.n]]
        des = np.concatenate([d for _, d in self.fs[:self.n]])
        self.owners = np.concatenate([np.full(len(d), i, np.int32)
            for i, (_, d) in enumerate(self.fs[:self.n])])
        self.global_matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=48))
        self.global_matcher.add([des]); self.global_matcher.train()
        saved = cache / 'visual_words.npy'
        if saved.exists():
            self.hist = np.load(saved)
        else:
            all_des = np.concatenate([d for _, d in self.fs])
            rng = np.random.default_rng(178)
            sample = all_des[rng.choice(len(all_des), min(70000, len(all_des)), replace=False)]
            km = MiniBatchKMeans(n_clusters=384, batch_size=2048, max_iter=40,
                n_init=1, random_state=178).fit(sample)
            mat = matcher(km.cluster_centers_.astype(np.float32), checks=24)
            hist = []
            for _, d in self.fs:
                hist.append(np.bincount([m.trainIdx for m in mat.match(d)], minlength=384))
            hist = np.array(hist, np.float32)
            idf = np.log((1+len(hist)) / (1+(hist > 0).sum(0)))
            hist = np.sqrt(hist) * idf
            hist /= np.maximum(np.linalg.norm(hist, axis=1, keepdims=True), 1e-8)
            self.hist = hist
            np.save(saved, hist)
        self.points = {}
        with open(data / 'train_points.csv', newline='') as f:
            for r in csv.DictReader(f):
                self.points.setdefault(r['image'], []).append((float(r['x']), float(r['y'])))

    def geometry(self, ti, p, d):
        pp, _ = self.fs[ti]
        pairs = self.local[ti].knnMatch(d, k=2)
        good = [m for pair in pairs if len(pair) == 2
                for m, n in [pair] if m.distance < .7*n.distance]
        if len(good) < 40:
            return None
        a = np.float32([pp[m.trainIdx] for m in good])
        b = np.float32([p[m.queryIdx] for m in good])
        h, mask = cv2.findHomography(a, b, cv2.RANSAC, 2)
        if h is None or mask is None or mask.sum() < 40 or mask.mean() < .35:
            return None
        if cv2.contourArea(cv2.convexHull(b[mask.ravel().astype(bool)])) < 6000:
            return None
        return h, int(mask.sum())

    def find(self, query_index):
        qi = self.n + query_index
        original = cv2.imread(str(self.paths[qi]), 0)
        height, width = original.shape
        options = {}
        for flipped in [False, True]:
            if flipped:
                p, d = root_sift(cv2.flip(original, 1), self.sift)
                p[:, 0] = width - 1 - p[:, 0]
            else:
                p, d = self.fs[qi]
            raw = self.global_matcher.knnMatch(d, k=2)
            good = [m for pair in raw if len(pair) == 2
                    for m, n in [pair] if m.distance < .85*n.distance]
            votes = np.bincount([self.owners[m.trainIdx] for m in good], minlength=self.n)
            candidates = set(int(i) for i in np.argsort(votes)[-15:] if votes[i] >= 3)
            if not flipped:
                sim = self.hist[:self.n] @ self.hist[qi]
                candidates.update(int(i) for i in np.argsort(sim)[-20:])
            for ti in sorted(candidates):
                geometry = self.geometry(ti, p, d)
                if geometry is None:
                    continue
                h, n = geometry
                if ti not in options or n > options[ti][1]:
                    options[ti] = h, n
        accepted = []
        for ti, (h, n) in options.items():
            donor = cv2.imread(str(self.train[ti]), 0)
            mask = cv2.warpPerspective(np.ones(donor.shape, np.uint8), h,
                (width, height), flags=cv2.INTER_NEAREST)
            coverage = float(mask.mean())
            if coverage < .3:
                continue
            warp = cv2.warpPerspective(donor, h, (width, height))
            selected = cv2.erode(mask, np.ones((9, 9), np.uint8)).astype(bool)
            corr = float(np.corrcoef(warp[selected], original[selected])[0, 1])
            if not np.isfinite(corr) or corr < .4:
                continue
            points = np.float32(self.points.get(self.train[ti].name, [])).reshape(-1, 1, 2)
            count = 0
            if len(points):
                xy = cv2.perspectiveTransform(points, h).reshape(-1, 2)
                count = int(((xy[:, 0] >= 0) & (xy[:, 0] < width)
                    & (xy[:, 1] >= 0) & (xy[:, 1] < height)).sum())
            score = coverage * corr * np.sqrt(n)
            accepted.append((score, self.train[ti].name, count, mask, coverage, corr, n))
        return max(accepted, key=lambda r: r[0]) if accepted else None
