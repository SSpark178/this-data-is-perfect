import os
os.environ['OMP_THREAD_LIMIT']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import cv2,numpy as np,json,subprocess,io,csv,concurrent.futures,time
from pathlib import Path
from features import stamps_on
cv2.setNumThreads(1)

def groups(im):
 out=[]
 for sx,sy,sw,sh in stamps_on(im):
  crop=im[sy+10:sy+sh-10,sx+10:sx+sw-10];gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
  for channel in [gray,crop[:,:,0],crop[:,:,1],crop[:,:,2]]:
   for inv in [False,True]:
    typ=cv2.THRESH_BINARY_INV if not inv else cv2.THRESH_BINARY;_,mask=cv2.threshold(channel,0,255,typ+cv2.THRESH_OTSU);mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((2,2),np.uint8));_,_,stats,_=cv2.connectedComponentsWithStats(mask);boxes=[tuple(map(int,b[:4]))for b in stats[1:]if 20<=b[3]<=165 and 6<=b[2]<=150 and .12<b[2]/b[3]<1.45 and b[4]>30 and b[4]/(b[2]*b[3])>.08]
    parents=list(range(len(boxes)))
    def find(i):
     while parents[i]!=i:parents[i]=parents[parents[i]];i=parents[i]
     return i
    for i,(x,y,w,h)in enumerate(boxes):
     for j,(xx,yy,ww,hh)in enumerate(boxes[:i]):
      gap=max(x,xx)-min(x+w,xx+ww)
      if min(h,hh)/max(h,hh)>.65 and abs(y+h/2-yy-hh/2)<max(h,hh)*.2 and -.2*min(w,ww)<gap<.45*max(h,hh):parents[find(i)]=find(j)
    clusters={}
    for i,b in enumerate(boxes):clusters.setdefault(find(i),[]).append(b)
    for bs in clusters.values():
     if not(2<=len(bs)<=4):continue
     bs.sort();x=min(b[0]for b in bs);y=min(b[1]for b in bs);xx=max(b[0]+b[2]for b in bs);yy=max(b[1]+b[3]for b in bs)
     if xx-x>360 or yy-y>165:continue
     globalbox=(x+sx+10,y+sy+10,xx-x,yy-y)
     if any(abs(globalbox[0]-o[0][0])<8 and abs(globalbox[1]-o[0][1])<8 and abs(globalbox[2]-o[0][2])<10 and abs(globalbox[3]-o[0][3])<10 for o in out):continue
     binary=255-mask[max(0,y-4):min(mask.shape[0],yy+4),max(0,x-4):min(mask.shape[1],xx+4)];binary=cv2.copyMakeBorder(binary,8,8,8,8,cv2.BORDER_CONSTANT,value=255);binary=cv2.resize(binary,None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST);out.append((globalbox,[(b[0]+sx+10,b[1]+sy+10,b[2],b[3])for b in bs],binary))
 return out

def process(p):
 name=p.stem if p.parent.name=='exhibits'else p.parent.name+'_'+p.stem;target=ROOT/(name+'.json');target.parent.mkdir(exist_ok=True)
 if target.exists():return name
 im=cv2.imread(str(p));out=[]
 for box,chars,binary in groups(im):
  data=cv2.imencode('.png',binary)[1].tobytes();cmd=['tesseract','stdin','stdout','--psm','7','-l','eng','-c','tessedit_char_whitelist=0123456789','tsv'];res=subprocess.run(cmd,input=data,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,check=True).stdout.decode();tokens=[r for r in csv.DictReader(io.StringIO(res),delimiter='\t')if r['text'].strip().isdigit()]
  if not tokens:continue
  text=''.join(r['text'].strip()for r in tokens);conf=np.mean([float(r['conf'])for r in tokens])
  if len(text)!=len(chars):continue
  out.append({'text':text,'confidence':float(conf),'box':box,'chars':chars})
 target.write_text(json.dumps(out));return name

# Repeated nonzero glyphs are an empirical cue, not proof of manipulation.
import cv2,numpy as np,json
from pathlib import Path

def bitmap(im,box):
 x,y,w,h=box;gray=cv2.cvtColor(im[max(0,y-1):y+h+1,max(0,x-1):x+w+1],cv2.COLOR_BGR2GRAY);_,m=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
 if(m>0).mean()>.6:m=255-m
 m=cv2.resize(m,(32,48),interpolation=cv2.INTER_AREA).astype(float).ravel()/255;return m/max(np.linalg.norm(m),1e-8)

def clone_proposals(im,name,cache):
 path=cache/(name+'.json')
 if not path.exists():return []
 rows=json.loads(path.read_text());out=[]
 for r in rows:
  text=r['text'];chars=r['chars'];conf=r['confidence']
  if r['box'][3]<25 or len(text)>4:continue
  for j in range(1,min(3,len(chars))):
   if text[j]=='0':continue
   for i in range(j):
    if text[i]=='0':continue
    sim=float(bitmap(im,chars[i])@bitmap(im,chars[j]));x,y,w,h=chars[j]
    if sim>.84:out.append((sim,conf,(x,y,w,h),text,i,j))
 return sorted(out,key=lambda r:-(r[0]+min(1,max(0,r[1]/100))*.1))
