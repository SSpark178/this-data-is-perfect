import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import cv2,numpy as np,joblib,time
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import KFold
from features import stamps_on, features_from_noise
from noiseprint import load_noiseprint, extract_noise
cv2.setNumThreads(1)

def proposals(pb,pn,valid,box):
 p=np.sqrt(pb*pn);small=cv2.boxFilter(p,-1,(9,9));large=cv2.boxFilter(p,-1,(35,35));score=small-.7*large;score[~valid]=-1;points=[]
 for _ in range(12):
  cy,cx=np.unravel_index(np.argmax(score),score.shape);points.append((cx,cy));cv2.circle(score,(cx,cy),8,-1,-1)
 for z in [pb,pn]:
  z=cv2.boxFilter(z,-1,(9,9));z[~valid]=-1;cy,cx=np.unravel_index(np.argmax(z),z.shape)
  if all(np.hypot(cx-x,cy-y)>5 for x,y in points):points.append((cx,cy))
 arrays=[];globals=[]
 for z in [pb,pn,p]:
  arrays.append([cv2.boxFilter(z,-1,(k,k))for k in [3,9,17,35]])
  vals=z[valid];globals.append([float(vals.mean()),float(vals.std()),*np.quantile(vals,[.5,.9,.99]).tolist()])
 rows=[];x,y,w,h=box
 for cx,cy in points:
  vec=[]
  for z,maps,gg in zip([pb,pn,p],arrays,globals):
   vals=[float(a[cy,cx])for a in maps];vec.extend(vals);vec.extend(gg);vec.extend([vals[1]-vals[3],(vals[1]-gg[0])/(gg[1]+.05)])
   for t in [.3,.5,.7]:
    selected=(z>max(t,float(z[cy,cx])*.7)).astype(np.uint8);n,lab,stats,_=cv2.connectedComponentsWithStats(selected);ind=lab[cy,cx];area=stats[ind,cv2.CC_STAT_AREA]if ind else 0;vec.append(np.log1p(area))
  vec.extend([(cx*3-x)/w,(cy*3-y)/h,min(cx*3-x,x+w-cx*3)/w,min(cy*3-y,y+h-cy*3)/h])
  rows.append((cx,cy,np.array(vec,np.float32)))
 return rows

def train_ranker(data, cache):
 ROOT = cache
 result = cache / "region_ranker.joblib"
 if result.exists(): return joblib.load(result)
 paths=sorted((data/'exhibits').glob('*.jpg'));rng=np.random.default_rng(178);xs=[];ys=[];groups=[];fs=[];masks=[];valids=[];boxes=[]
 for i,path in enumerate(paths):
  im=cv2.imread(str(path));f=features_from_noise(im,np.load(cache / ('training_noise_'+path.stem+'.npy')));mask=cv2.imread(str(path.with_name(path.stem+'_mask.png')),0)>0;valid=np.zeros(mask.shape,np.uint8);bb=stamps_on(im)
  for x,y,w,h in bb:valid[y+15:y+h-15,x+15:x+w-15]=1
  box=max(bb,key=lambda r:r[2]*r[3]);v=valid[::3,::3].astype(bool);lab=mask[::3,::3];pos=np.flatnonzero(lab.ravel()&v.ravel());neg=np.flatnonzero(~lab.ravel()&v.ravel());neg=rng.choice(neg,min(3500,len(neg)),replace=False);take=np.r_[pos,neg];xs.append(f.reshape(-1,f.shape[-1])[take]);ys.append(lab.ravel()[take]);groups.extend([i]*len(take));fs.append(f);masks.append(mask);valids.append(v);boxes.append(box)
 x=np.concatenate(xs);y=np.concatenate(ys);groups=np.array(groups);maps={};start=time.monotonic()
 cache=ROOT/'oof_regions.joblib'
 if cache.exists():maps=joblib.load(cache)
 else:
  for fold,(train,test)in enumerate(KFold(5,shuffle=True,random_state=178).split(paths)):
   ps={i:[]for i in test};sel=np.isin(groups,train)
   for dim in [85,142]:
    model=ExtraTreesClassifier(n_estimators=48,max_depth=24,min_samples_leaf=5,max_features=.8,n_jobs=2,random_state=178,class_weight='balanced').fit(x[sel,:dim],y[sel])
    for i in test:
     f=fs[i][:,:,:dim];ps[i].append(model.predict_proba(f.reshape(-1,dim))[:,1].reshape(f.shape[:2]).astype(np.float32))
   for i in test:maps[int(i)]=ps[i]
   print('fold',fold,'seconds',round(time.monotonic()-start),flush=True)
  joblib.dump(maps,cache)
 samples=[];targets=[];gs=[];raw=[]
 for i in range(30):
  rows=proposals(*maps[i],valids[i],boxes[i]);true=0
  for cx,cy,vec in rows:
   yy=min(masks[i].shape[0]-1,cy*3);xx=min(masks[i].shape[1]-1,cx*3);hit=bool(masks[i][yy,xx]);samples.append(vec);targets.append(hit);gs.append(i);raw.append((i,cx,cy));true+=hit
  print(paths[i].name,'positives',true,'/',len(rows),flush=True)
 samples=np.array(samples);targets=np.array(targets);gs=np.array(gs);hits=[]
 for train,test in KFold(5,shuffle=True,random_state=178).split(paths):
  model=ExtraTreesClassifier(n_estimators=300,min_samples_leaf=3,max_depth=8,max_features=.8,class_weight='balanced',random_state=178,n_jobs=2).fit(samples[np.isin(gs,train)],targets[np.isin(gs,train)])
  for i in test:
   ids=np.flatnonzero(gs==i);pp=model.predict_proba(samples[ids])[:,1];best=ids[np.argmax(pp)];hits.append(int(targets[best]));print('stack val',paths[i].name,int(targets[best]),flush=True)
 print('STACK HITS',sum(hits),len(hits),flush=True)
 model=ExtraTreesClassifier(n_estimators=300,min_samples_leaf=3,max_depth=8,max_features=.8,class_weight='balanced',random_state=178,n_jobs=2).fit(samples,targets);joblib.dump(model,ROOT/'region_ranker.joblib',compress=3)

 return model
