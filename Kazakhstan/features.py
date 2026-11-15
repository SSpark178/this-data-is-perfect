import cv2
import numpy as np

def stamps_on(page):
 d=np.abs(page.astype(np.int16)-np.array([38,36,34])).max(2)
 m=cv2.morphologyEx((d>30).astype(np.uint8),cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))
 _,_,stats,_=cv2.connectedComponentsWithStats(m)
 return [(int(x),int(y),int(w),int(h))for x,y,w,h,area in stats[1:]if w>120 and h>120]


def rgb_features(im,stride=3):
 a=im.astype(np.float32);g=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY).astype(np.float32);lab=cv2.cvtColor(im,cv2.COLOR_BGR2Lab).astype(np.float32)
 maps=[g/255,lab[:,:,1]/255,lab[:,:,2]/255]
 def add(z):maps.append(z[::stride,::stride].copy())
 maps=[m[::stride,::stride].copy()for m in maps]
 for k in [3,7,15,31]:
  mu=cv2.boxFilter(g,-1,(k,k));var=np.maximum(cv2.boxFilter(g*g,-1,(k,k))-mu*mu,0)
  add(mu/255);add(np.sqrt(var)/30)
  for channel in [1,2]:
   z=lab[:,:,channel];m=cv2.boxFilter(z,-1,(k,k));v=np.maximum(cv2.boxFilter(z*z,-1,(k,k))-m*m,0);add(np.sqrt(v)/20)
 med=g-cv2.medianBlur(im[:,:,0],3).astype(np.float32)
 residuals=[g-cv2.medianBlur(g,3),g-cv2.GaussianBlur(g,(0,0),.6),g-cv2.GaussianBlur(g,(0,0),1.2),cv2.Laplacian(g,-1)/4,lab[:,:,1]-cv2.GaussianBlur(lab[:,:,1],(0,0),.7),lab[:,:,2]-cv2.GaussianBlur(lab[:,:,2],(0,0),.7)]
 for r in residuals:
  ar=np.abs(r)
  for k in [5,13,29]:
   mu=cv2.boxFilter(r,-1,(k,k));v=np.maximum(cv2.boxFilter(r*r,-1,(k,k))-mu*mu,0)
   add(cv2.boxFilter(ar,-1,(k,k))/12);add(np.sqrt(v)/12)
   add(cv2.boxFilter((ar<.5).astype(np.float32),-1,(k,k)))
 for quality in [75,85,90,95]:
  again=cv2.imdecode(cv2.imencode('.jpg',im,[cv2.IMWRITE_JPEG_QUALITY,quality])[1],1).astype(np.float32)
  diff=np.abs(a-again).mean(2)
  for k in [7,19,39]:add(cv2.boxFilter(diff,-1,(k,k))/10)
 return np.stack(maps,axis=2)


def features_from_noise(im,n,stride=3):
 out=[]
 def add(z):out.append(z[::stride,::stride].copy())
 add(n)
 valid=np.zeros(n.shape,bool)
 for x,y,w,h in stamps_on(im):valid[y+18:y+h-18,x+18:x+w-18]=True
 for period in [4,8,16]:
  pattern=np.zeros(n.shape,np.float32)
  for yy in range(period):
   for xx in range(period):
    vals=n[yy::period,xx::period];vv=valid[yy::period,xx::period];mean=float(np.median(vals[vv]))if vv.any()else 0;pattern[yy::period,xx::period]=mean
  r=n-pattern
  for k in [7,15,31,63]:
   mu=cv2.boxFilter(r,-1,(k,k));var=np.maximum(cv2.boxFilter(r*r,-1,(k,k))-mu*mu,0);add(mu);add(np.sqrt(var));add(cv2.boxFilter(np.abs(r),-1,(k,k)));cov=cv2.boxFilter((n-n.mean())*(pattern-pattern.mean()),-1,(k,k));add(cov)
 for k in [7,15,31,63]:
  mu=cv2.boxFilter(n,-1,(k,k));var=np.maximum(cv2.boxFilter(n*n,-1,(k,k))-mu*mu,0);add(mu);add(np.sqrt(var))
 return np.concatenate([rgb_features(im,stride),np.stack(out,axis=2)],axis=2)
