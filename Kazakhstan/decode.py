import csv
import string
import cv2
import numpy as np
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

def templates():
 out={}
 for letter in string.ascii_uppercase:
  a=np.zeros((300,300),np.uint8);cv2.putText(a,letter,(20,250),cv2.FONT_HERSHEY_SIMPLEX,7,255,1,cv2.LINE_8);a=skeletonize(a>0).astype(np.uint8);yy,xx=np.where(a>0);x0,x1=xx.min(),xx.max();y0,y1=yy.min(),yy.max();pts=np.array([((x-x0)/(x1-x0) if x1>x0 else .5,(y-y0)/(y1-y0))for x,y in zip(xx,yy)]);tree=cKDTree(pts);length=0
  for y,x in zip(yy,xx):
   for dy,dx in [(0,1),(1,0),(1,1),(1,-1)]:
    if not(0<=y+dy<300 and 0<=x+dx<300)or not a[y+dy,x+dx]:continue
    if dx and dy and(a[y,x+dx]or a[y+dy,x]):continue
    length+=np.hypot(dx/max(1,x1-x0),dy/(y1-y0))
  out[letter]=(tree,max(length,1))
 return out

def candidates(folder):
 albums=[]
 for ai in range(1,6):
  pages=[]
  for file in sorted((folder/('album_'+str(ai))).glob('*.npz')):
   q=np.load(file)['prob'];s=cv2.boxFilter(q,-1,(7,7))-.4*cv2.boxFilter(q,-1,(39,39));s[q==0]=-1;pts=[];ws=[]
   for k in range(8):
    y,x=np.unravel_index(s.argmax(),s.shape);pts.append((x*3,y*3));ws.append(s[y,x]);cv2.circle(s,(x,y),12,-1,-1)
   ws=np.array(ws);weights=np.exp((ws-ws.max())/.08);weights/=weights.sum();pages.append((np.array(pts),weights))
  albums.append(pages)
 return albums

def scores(pages,ts,frame,sigma=.065):
 x,y,w,h=frame;p=np.array([r[0]for r in pages]);q=np.array([r[1]for r in pages]);p=(p-np.array([x,y]))/np.array([w,h]);flat=p.reshape(-1,2);out={}
 for letter,(tree,length) in ts.items():
  d=tree.query(flat)[0].reshape(p.shape[:2]);mass=(np.exp(-.5*(d/sigma)**2)*q).sum(1)/(np.sqrt(2*np.pi)*sigma*length);mass *= (1600*1200)/(w*h)
  out[letter]=float(np.log(.35+.65*mass).sum())
 return out

def decode(fields, output):
    albums = candidates(fields)
    if len(albums) != 5 or any(len(pages) != 26 for pages in albums):
        raise ValueError('Expected five albums of 26 maps each')
    shapes = templates()
    fits = []
    # Search every uppercase letter independently; no word list is used.
    for x in [350, 450, 550, 650]:
        for y in [150, 250, 350]:
            for w in [400, 550, 700, 850]:
                for h in [500, 650, 800, 950]:
                    if x+w > 1500 or y+h > 1170:
                        continue
                    frame = x, y, w, h
                    ss = [scores(p, shapes, frame, .065) for p in albums]
                    chosen = [max(s, key=s.get) for s in ss]
                    total = sum(s[l] for s, l in zip(ss, chosen))
                    fits.append((total, ''.join(chosen), frame, ss))
    fits.sort(key=lambda r: -r[0])
    total, word, frame, ss = fits[0]
    x, y, w, h = frame
    raw, selected = [], []
    for ai, pages in enumerate(albums):
        paths = sorted((fields/f'album_{ai+1}').glob('*.npz'))
        tree, _ = shapes[word[ai]]
        for path, (points, weights) in zip(paths, pages):
            distances = tree.query((points-[x, y])/[w, h])[0]
            joint = weights*np.exp(-.5*(distances/.065)**2)
            rank = int(np.argmax(joint))
            for collection, point in [(raw, points[0]), (selected, points[rank])]:
                cx, cy = map(int, point)
                collection.append((path.parent.name, path.stem, cx-30, cy-25, 60, 50))
    for name, rows in [('raw_boxes.csv', raw), ('boxes.csv', selected)]:
        with (output/name).open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['album', 'page', 'x', 'y', 'w', 'h'])
            writer.writerows(rows)
    with (output/'decode_scores.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['album', 'letter', 'score', 'selected'])
        for i, s in enumerate(ss):
            for letter, score in sorted(s.items(), key=lambda r: -r[1]):
                writer.writerow([f'album_{i+1}', letter, score, int(letter == word[i])])
    with (output/'frame_scores.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['score', 'letters', 'x', 'y', 'w', 'h'])
        for score, letters, f, _ in fits[:15]:
            writer.writerow([score, letters, *f])
    (output/'flag.txt').write_text('SIGMOID_'+word+'\n')
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="610" viewBox="0 0 1600 610">',
           '<rect width="1600" height="610" fill="white"/>']
    for ri, (label, rows) in enumerate([('Raw forensic peaks', raw), ('Joint forensic and letter fit', selected)]):
        base_y = ri*300
        svg.append(f'<text x="12" y="{base_y+20}" font-size="16">{label}</text>')
        for i in range(5):
            svg.append(f'<text x="{i*320+12}" y="{base_y+45}" font-size="15">album_{i+1}: {word[i]}</text>')
        for album, page, bx, by, bw, bh in rows:
            ai = int(album[-1])-1
            cx = ai*320+(bx+bw/2)/5
            cy = base_y+55+(by+bh/2)/5
            svg.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="3.2" fill="#202020"/>')
    svg.append('</svg>')
    (output/'overlay.svg').write_text('\n'.join(svg))
    print('SIGMOID_'+word, flush=True)
    return word
