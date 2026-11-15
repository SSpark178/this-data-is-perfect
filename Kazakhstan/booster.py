import cv2
import numpy as np
from scipy.ndimage import uniform_filter
from lightgbm import LGBMClassifier
from features import features_from_noise, stamps_on
from extra_features import features as extra

def forensic_features(im,n):
 f=features_from_noise(im,n);e=extra(im)[:,:,85:];v=np.zeros(im.shape[:2],np.float32)
 for x,y,w,h in stamps_on(im):v[y+15:y+h-15,x+15:x+w-15]=1
 v=v[::3,::3];out=[f,e]
 for k in [15,35]:
  den=cv2.boxFilter(v,-1,(k,k))+1e-5;mu=uniform_filter(f*v[:,:,None],size=(k,k,1),mode="mirror")/den[:,:,None];out.append(f-mu)
 return np.concatenate(out,axis=2)
def model():return LGBMClassifier(n_estimators=250,num_leaves=15,max_depth=-1,learning_rate=.04,min_child_samples=100,colsample_bytree=.8,reg_lambda=3,class_weight='balanced',n_jobs=2,verbosity=-1,random_state=178)
