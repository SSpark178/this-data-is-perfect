import cv2,numpy as np
from features import rgb_features as base_features

def features(im,stride=3):
 base=base_features(im,stride);a=im.astype(np.float32);gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY).astype(np.float32);r=gray-cv2.GaussianBlur(gray,(0,0),.7);rgb=a-cv2.GaussianBlur(a,(0,0),.7);out=[]
 def add(z):out.append(z[::stride,::stride].copy())
 for k in [7,15,31]:
  rr=cv2.boxFilter(r*r,-1,(k,k))
  for dy,dx in [(0,1),(1,0),(1,1),(1,-1),(0,2),(2,0),(2,2),(0,3)]:
   shifted=np.roll(r,(dy,dx),(0,1));cov=cv2.boxFilter(r*shifted,-1,(k,k));ss=cv2.boxFilter(shifted*shifted,-1,(k,k));add(cov/np.sqrt((rr+.02)*(ss+.02)))
  vv=[cv2.boxFilter(rgb[:,:,c]*rgb[:,:,c],-1,(k,k))for c in range(3)]
  for c,d in [(0,1),(0,2),(1,2)]:
   cov=cv2.boxFilter(rgb[:,:,c]*rgb[:,:,d],-1,(k,k));add(cov/np.sqrt((vv[c]+.02)*(vv[d]+.02)))
  total=sum(vv);add(np.log1p(total));add(np.minimum.reduce(vv)/(total+.01))
  for c in range(3):add(vv[c]/(total+.01))
  # Image-content-normalized high frequency noise.
  dx=cv2.Sobel(gray,cv2.CV_32F,1,0);dy=cv2.Sobel(gray,cv2.CV_32F,0,1);grad=cv2.boxFilter(dx*dx+dy*dy,-1,(k,k));add(np.log1p(rr)/np.log(2+grad))
 return np.concatenate([base,np.stack(out,axis=2)],axis=2)
