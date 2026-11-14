import csv
import zipfile
from functools import lru_cache

import cv2
import numpy as np

cv2.setNumThreads(1)
cv2.setRNGSeed(0)


def normalized(a):
    a = a.astype(np.float32).ravel()
    a -= a.mean()
    return a / max(float(np.linalg.norm(a)), 1e-8)


def main():
    with open('output.csv', 'w', newline='') as f:
        csv.writer(f).writerow(['image_a', 'image_b'])
    with zipfile.ZipFile('test-input.zip') as archive:
        names = sorted(n for n in archive.namelist()
                       if not n.endswith('/') and not n.startswith('__MACOSX/')
                       and n.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff')))
        def decode(name):
            raw = archive.read(name)
            flag = cv2.IMREAD_GRAYSCALE
            if raw[:2] == b'\xff\xd8':
                if len(raw) > 4000000:
                    flag = cv2.IMREAD_REDUCED_GRAYSCALE_4
                elif len(raw) > 1000000:
                    flag = cv2.IMREAD_REDUCED_GRAYSCALE_2
            return cv2.imdecode(np.frombuffer(raw, np.uint8), flag)

        vectors, hashes, usable = [], [], []
        for name in names:
            im = decode(name)
            if im is None:
                continue
            usable.append(name)
            vectors.append(normalized(cv2.resize(im, (32, 32), interpolation=cv2.INTER_AREA)))
            d = cv2.dct(cv2.resize(im, (32, 32)).astype(np.float32))[:8, :8].ravel()[1:]
            hashes.append(d > np.median(d))
        names = usable
        if len(names) < 2:
            return
        vectors = np.stack(vectors)
        hashes = np.stack(hashes)
        candidates = set()
        # Search using compact hashes; avoid repeated large matrix products.
        packed = np.packbits(hashes, axis=1)
        bitcount = np.array([int(k).bit_count() for k in range(256)], dtype=np.uint8)
        for i in range(len(names)):
            distances = bitcount[np.bitwise_xor(packed, packed[i])].sum(axis=1)
            distances[i] = 255
            closest = np.argpartition(distances, min(2, len(names)-1))[:min(3, len(names)-1)]
            near = np.flatnonzero(distances <= 8)
            for j in np.concatenate((near, closest)):
                if i != j and distances[j] <= 26:
                    candidates.add((min(i, int(j)), max(i, int(j))))

        @lru_cache(maxsize=48)
        def features(i):
            im = decode(names[i])
            scale = min(1.0, 560.0 / max(im.shape))
            if scale < 1:
                im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            keypoints, descriptors = cv2.SIFT_create(nfeatures=650, contrastThreshold=0.025).detectAndCompute(im, None)
            points = np.array([k.pt for k in keypoints], dtype=np.float32)
            return im, points, descriptors

        matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=3), dict(checks=32))

        def duplicate(i, j):
            if float(vectors[i] @ vectors[j]) > 0.995:
                return True
            a, pa, da = features(i)
            b, pb, db = features(j)
            if da is None or db is None or min(len(da), len(db)) < 12:
                return False
            matches = matcher.knnMatch(da, db, k=2)
            good = [m for pair in matches if len(pair) == 2 for m, n in [pair] if m.distance < 0.70 * n.distance]
            if len(good) < 12:
                return False
            p = np.float32([pa[m.queryIdx] for m in good])
            q = np.float32([pb[m.trainIdx] for m in good])
            H, mask = cv2.findHomography(p, q, cv2.RANSAC, 2.5)
            if H is None or mask is None:
                return False
            keep = mask.ravel().astype(bool)
            count = int(keep.sum())
            if count < 12 or count / len(good) < 0.5:
                return False
            for pts, im in [(p[keep], a), (q[keep], b)]:
                if cv2.contourArea(cv2.convexHull(pts)) / im.size < 0.035:
                    return False
            warped = cv2.warpPerspective(a, H, (b.shape[1], b.shape[0]))
            valid = cv2.warpPerspective(np.ones_like(a), H, (b.shape[1], b.shape[0]), flags=cv2.INTER_NEAREST) > 0
            valid = cv2.erode(valid.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
            if valid.sum() < b.size * 0.15:
                return False
            aa = cv2.GaussianBlur(warped, (5, 5), 0)[valid]
            bb = cv2.GaussianBlur(b, (5, 5), 0)[valid]
            corr = float(normalized(aa) @ normalized(bb))
            return corr > 0.92 and (count >= 18 or corr > 0.97)

        pairs = [(names[i], names[j]) for i, j in sorted(candidates, key=lambda pair: (pair[0], pair[1])) if duplicate(i, j)]
        with open('output.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['image_a', 'image_b'])
            writer.writerows(pairs)


if __name__ == '__main__':
    main()
