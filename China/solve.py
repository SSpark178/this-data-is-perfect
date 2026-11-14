#!/usr/bin/env python3
"""Oracle-bone glyph counter.

Usage:
    python solve.py --data dataset --out counts.csv

The input dataset must contain:
  train/images/*.jpg, train_labels.csv, crates.csv
  crates/crate_XX/set_Y/*.jpg

HOG proposes glyph windows. A CNN classifies them. A second-stage classifier,
calibrated against held-out labelled rubbings, filters false detections.
Counts in all five sets form five bits, with A=1 ... Z=26.
"""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from xgboost import XGBClassifier

cv2.setNumThreads(1)
torch.set_num_threads(min(5, torch.get_num_threads()))
PATCH_SIZE = (32, 48)
HOG_SIZE = (40, 64)
DETECTION_THRESHOLD = .75


def crop(im, x, y, w, h, pad=.08):
    x0 = max(0, int(x - w * pad))
    y0 = max(0, int(y - h * pad))
    x1 = min(im.shape[1], int(x + w * (1 + pad)))
    y1 = min(im.shape[0], int(y + h * (1 + pad)))
    region = im[y0:y1, x0:x1]
    return cv2.resize(region, PATCH_SIZE) if region.size else None


def hog_descriptor():
    return cv2.HOGDescriptor(HOG_SIZE, (16, 16), (8, 8), (8, 8), 9)


def train_hog(data, labels, targets, train_names):
    feature_vectors, class_names = [], []
    descriptor = hog_descriptor()
    for name, group in labels.groupby("image"):
        if name not in train_names:
            continue
        im = cv2.imread(str(data / "train/images" / name), 0)
        if im is None:
            raise FileNotFoundError(name)
        for box in group.itertuples():
            r = im[int(box.y):int(box.y+box.h), int(box.x):int(box.x+box.w)]
            if r.size:
                feature_vectors.append(descriptor.compute(cv2.resize(r, HOG_SIZE)).ravel())
                class_names.append(box.glyph_id)
    X = np.asarray(feature_vectors)
    y = np.asarray(class_names)
    models = {}
    for glyph in targets:
        model = LinearSVC(C=.05, class_weight="balanced", dual=False, max_iter=2000)
        model.fit(X, (y == glyph).astype(int))
        models[glyph] = model
    return models


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 12, 3, padding=1), nn.BatchNorm2d(12), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(12, 24, 3, padding=1), nn.BatchNorm2d(24), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 40, 3, padding=1), nn.BatchNorm2d(40), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(40, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU())
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(48*4*6, 64), nn.ReLU(), nn.Dropout(.15),
            nn.Linear(64, 9))

    def forward(self, x):
        return self.head(self.features(x))


def train_cnn(data, labels, targets, train_names, seed):
    rng = np.random.default_rng(seed)
    images, classes = [], []
    for name, group in labels.groupby("image"):
        if name not in train_names:
            continue
        im = cv2.imread(str(data / "train/images" / name), 0)
        if im is None:
            raise FileNotFoundError(name)
        for box in group.itertuples():
            label = targets.index(box.glyph_id) if box.glyph_id in targets else 8
            original = crop(im, box.x, box.y, box.w, box.h)
            if original is not None:
                images.append(original)
                classes.append(label)
            if label != 8:
                for _ in range(2):
                    dx, dy = rng.uniform(-.13, .13), rng.uniform(-.12, .12)
                    scale = rng.uniform(.85, 1.2)
                    augmented = crop(im, box.x + dx*box.w,
                                     box.y + dy*box.h, box.w*scale,
                                     box.h*scale, pad=.06)
                    if augmented is not None:
                        images.append(augmented)
                        classes.append(label)
        for _ in range(3):
            w = rng.uniform(40, 110)
            h = w*rng.uniform(1, 1.8)
            x = rng.uniform(0, max(1, im.shape[1]-w))
            y = rng.uniform(0, max(1, im.shape[0]-h))
            overlap = any(
                box.glyph_id in targets
                and abs(x+w/2-box.x-box.w/2) < .8*w+.5*box.w
                and abs(y+h/2-box.y-box.h/2) < .8*h+.5*box.h
                for box in group.itertuples())
            if not overlap:
                negative = crop(im, x, y, w, h, pad=0)
                if negative is not None:
                    images.append(negative)
                    classes.append(8)

    X = torch.tensor(np.asarray(images)[:, None], dtype=torch.float32)/255
    Y = torch.tensor(classes, dtype=torch.long)
    frequencies = np.bincount(classes, minlength=9)
    weights = [1/np.sqrt(frequencies[y]) for y in classes]
    sampler = WeightedRandomSampler(weights, 25000, replacement=True)
    batches = DataLoader(TensorDataset(X, Y), batch_size=256, sampler=sampler)
    net = Network()
    optimizer = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=1e-4)
    checkpoints = {}
    for epoch in range(1, 21):
        net.train()
        for a, b in batches:
            a = (a * torch.empty((len(a),1,1,1)).uniform_(.8,1.25)
                 + torch.empty((len(a),1,1,1)).uniform_(-.09,.09)).clamp(0,1)
            loss = nn.functional.cross_entropy(net(a), b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch in (10, 15, 20):
            checkpoints[epoch] = copy.deepcopy(net.state_dict())
        if epoch % 5 == 0:
            print("Trained CNN epoch", epoch, flush=True)
    return checkpoints


def nonmaximum_suppression(boxes):
    if not boxes:
        return []
    boxes = np.asarray(boxes, dtype=float)
    order = np.argsort(boxes[:,4])[::-1]
    selected = []
    while len(order):
        i = order[0]
        selected.append(boxes[i])
        candidates = boxes[order[1:]]
        dx = abs(candidates[:,0]+candidates[:,2]/2
                 - boxes[i,0]-boxes[i,2]/2)
        dy = abs(candidates[:,1]+candidates[:,3]/2
                 - boxes[i,1]-boxes[i,3]/2)
        near = (dx < np.minimum(boxes[i,2],candidates[:,2])*.35) & (
            dy < np.minimum(boxes[i,3],candidates[:,3])*.35)
        order = order[1:][~near]
    return selected


def detect(im, detector, scales):
    proposed = []
    for scale in scales:
        scaled = cv2.resize(im, None, fx=scale, fy=scale,
                            interpolation=cv2.INTER_AREA
                            if scale < 1 else cv2.INTER_LINEAR)
        if min(scaled.shape) < 65:
            continue
        positions, scores = detector.detect(
            scaled, hitThreshold=.25, winStride=(8,8), padding=(0,0))
        for (x,y),score in zip(positions,scores):
            proposed.append([x/scale,y/scale,40/scale,64/scale,float(score)])
    return nonmaximum_suppression(proposed)


def generate_candidates(data, labels, crates, hog_models, validation_names):
    known = {(name, g): group[["x","y","w","h"]].to_numpy()
             for (name,g),group in labels.groupby(["image","glyph_id"])}
    val, unseen = [], []
    for crate,glyph in zip(crates.crate,crates.glyph_id):
        detector = hog_descriptor()
        svm = hog_models[glyph]
        detector.setSVMDetector(
            np.r_[svm.coef_[0],svm.intercept_[0]].astype("float32"))
        sizes = labels[labels.glyph_id==glyph]
        base = .5*(40/sizes.w.median()+64/sizes.h.median())
        scales = [base*ratio for ratio in (.75,1,1.33)]
        jobs = [(data/"train/images"/name,0,name)
                for name in sorted(validation_names)]
        for set_number in range(1,6):
            directory = data/"crates"/crate/("set_"+str(set_number))
            jobs += [(p,set_number,p.name) for p in sorted(directory.glob("*.jpg"))]

        def process(item):
            path,set_number,name = item
            im = cv2.imread(str(path),0)
            if im is None:
                raise FileNotFoundError(path)
            return [(str(path),crate,set_number,glyph,*b,name)
                    for b in detect(im,detector,scales)]

        with ThreadPoolExecutor(max_workers=4) as pool:
            groups = list(pool.map(process,jobs))
        for group in groups:
            for path,cr,num,g,x,y,w,h,score,name in group:
                row = [path,cr,num,g,x,y,w,h,score]
                if num:
                    unseen.append(row)
                else:
                    gt = known.get((name,g),np.empty((0,4)))
                    cx,cy = x+w/2,y+h/2
                    matched = any(gx-.2*gw<cx<gx+1.2*gw
                                  and gy-.2*gh<cy<gy+1.2*gh
                                  for gx,gy,gw,gh in gt)
                    val.append(row+[int(matched)])
        print("Proposals",crate,flush=True)
    cols=["image","crate","set","glyph","x","y","w","h","score"]
    return (pd.DataFrame(val,columns=cols+["truth"]),
            pd.DataFrame(unseen,columns=cols))


def score_candidates(df,models,targets):
    df = df.reset_index(drop=True).copy()
    samples = np.empty((len(df),48,32),dtype=np.uint8)
    for image,rows in df.groupby("image"):
        im = cv2.imread(image,0)
        if im is None:
            raise FileNotFoundError(image)
        for i,r in rows.iterrows():
            v = crop(im,r.x,r.y,r.w,r.h)
            samples[i] = v if v is not None else 128
    tensor = torch.tensor(samples[:,None],dtype=torch.float32)/255
    for ep,parameters in sorted(models.items()):
        model = Network()
        model.load_state_dict(parameters)
        model.eval()
        predictions=[]
        with torch.no_grad():
            for batch in tensor.split(512):
                predictions.append(model(batch).softmax(-1).numpy())
        p = np.concatenate(predictions)
        df["p"+str(ep)] = [p[i,targets.index(g)]
                           for i,g in enumerate(df.glyph)]
    df["ratio"]=df.w/df.h
    return df


def decode(values):
    """Separate low/high counts into two levels with a singleton penalty."""
    ordered = np.sort(np.asarray(values,dtype=float))
    splits=[]
    for nlow in range(1,5):
        low,high=ordered[:nlow],ordered[nlow:]
        error = np.square(low-low.mean()).sum()+np.square(high-high.mean()).sum()
        if len(high)==1:
            error+=12.0
        splits.append((error,(ordered[nlow-1]+ordered[nlow])/2))
    threshold=min(splits)[1]
    bits="".join("1" if x>threshold else "0" for x in values)
    v=int(bits,2)
    return bits,chr(64+v) if 1<=v<=26 else "?"


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--data",type=Path,default=Path("dataset"))
    parser.add_argument("--out",type=Path,default=Path("counts.csv"))
    parser.add_argument("--seed",type=int,default=42)
    args=parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    labels=pd.read_csv(args.data/"train_labels.csv")
    crates=pd.read_csv(args.data/"crates.csv")
    targets=list(crates.glyph_id)
    train_names,valid_names=train_test_split(
        np.asarray(labels.image.unique()),test_size=.22,random_state=42)
    H=train_hog(args.data,labels,targets,set(train_names))
    cnn=train_cnn(args.data,labels,targets,set(train_names),args.seed)
    val,images=generate_candidates(args.data,labels,crates,H,set(valid_names))
    val=score_candidates(val,cnn,targets)
    images=score_candidates(images,cnn,targets)
    features=["score","p10","p15","p20","w","h","ratio"]
    output=[]
    for crate,glyph in zip(crates.crate,crates.glyph_id):
        a=val[val.crate==crate]
        b=images[images.crate==crate]
        truth=a.truth.astype(int)
        weight=min(100,max(1,int((len(a)-truth.sum()) /
                                  max(1,truth.sum())/5)))
        model=XGBClassifier(n_estimators=200,max_depth=3,learning_rate=.05,
                            subsample=.7,colsample_bytree=.9,min_child_weight=2,
                            reg_lambda=10,tree_method="hist",n_jobs=4,
                            objective="binary:logistic",eval_metric="aucpr",
                            scale_pos_weight=weight,random_state=args.seed)
        model.fit(a[features],truth)
        confidence=model.predict_proba(b[features])[:,1]
        positive=b.assign(strong=(confidence>=DETECTION_THRESHOLD).astype(int))
        counts=positive.groupby("set").strong.sum().reindex(
            range(1,6),fill_value=0).to_numpy(dtype=int)
        bits,letter=decode(counts)
        print(crate,glyph,counts.tolist(),bits,letter,flush=True)
        for i,count in enumerate(counts,1):
            output.append([crate,"set_"+str(i),glyph,int(count)])
    result=pd.DataFrame(output,columns=["crate","set","glyph_id","count"])
    args.out.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(args.out,index=False)
    letters=[]
    for crate in crates.crate:
        c=result[result.crate==crate].sort_values("set")["count"]
        letters.append(decode(c)[1])
    print("FLAG: SIGMOID_"+"".join(letters))
    print("Written:",args.out)


if __name__=="__main__":
    main()
