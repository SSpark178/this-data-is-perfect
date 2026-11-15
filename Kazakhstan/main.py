"""Run: python main.py --data /path/to/data --output result

RGB residual, ELA, Noiseprint++ and local context features feed a pixel
classifier trained on the supplied masks. Its out-of-fold development score
is 20/30 region centres. Keep eight candidate regions per page, then fit the
five overlays jointly to the uppercase alphabet at a shared position/scale.
No dictionary, reference stamps or manual album annotations are used.
model.lgb stores the fitted 30-exhibit classifier; delete it to retrain.
Both raw detections and the joint selection are exported for inspection.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import argparse
from pathlib import Path
import cv2
import torch
from detector import detect
from decode import decode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('result'))
    parser.add_argument('--fields', type=Path, help='Reuse existing probability maps')
    parser.add_argument('--model', type=Path, default=Path(__file__).with_name('model.lgb'))
    args = parser.parse_args()
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    cache = args.output/'cache'
    cache.mkdir(exist_ok=True)
    fields = args.fields or args.output/'fields'
    detect(args.data, cache, fields, args.model)
    decode(fields, args.output)


if __name__ == '__main__':
    main()
