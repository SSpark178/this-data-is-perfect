import re,unicodedata,collections,itertools
import pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent
R=pd.read_csv(HERE / 'references.csv');P=pd.read_csv(HERE / 'panel.csv');refmap=R.groupby(['round','card']).reference.apply(list).to_dict()

def norm(t):
 t=t.lower().replace('’',"'")
 t=''.join(c for c in unicodedata.normalize('NFKD',t) if not unicodedata.combining(c))
 return t

CATS={
'cuisine':{
 'japanese':r'\bjapanese\b|\bsushi\b|\boriental\b',
 'italian':r'\bitalian\b|\bspaghetti\b|\bpasta\b',
 'french':r'\bfrench\b',
 'english':r'\benglish\b|\bbritish\b',
 'indian':r'\bindian\b|\bcurry\b',
 'chinese':r'\bchinese\b',
 'fast':r'\bfast[ -]?food\b|\bburger\b|\bfries\b|\bchips\b'
},
'etype':{
 'coffee':r'\bcoffee shop\b|\bcaf[eé]\b|\bcoffee house\b',
 'pub':r'\bpub\b|\bpublic house\b|\bbar\b(?! one)|\btavern\b',
 'restaurant':r'\brestaurant\b'
},
'price':{
 'cheap':r'\bcheap\b|\binexpensive\b|\blow[ -](?:priced?|cost|budget)\b|\baffordable\b|\bbudget\b|\bunder ?£ ?20\b|\bless than £ ?20\b|\bless than 20\b|\bprice range less than £?20\b',
 'moderate':r'\bmoderate(?:ly)?\b|\bmid[ -]range\b|\bmid[ -]priced\b|\bmedium[ -]priced\b|\baverage[ -]priced\b|\breasonab(?:le|ly) priced\b|£ ?20 ?[–-] ?£?25\b|\b20[ -]25\b|\bbetween £?20 and £?25\b',
 'high':r'\bhigh[ -](?:priced?|cost|budget)\b|\bexpensive\b|\bpricey\b|\bhigh price\b|\bmore than ?£ ?30\b|\bover £ ?30\b|\bgreater than ?£ ?30\b|\bpremium\b|\bupscale\b|\bbig spenders\b'
},
'rating':{
 'low':r'\b(?:1|one) (?:out of|of) (?:5|five)\b|\b(?:1|one)[ -]?star\b|\blow(?:ly)?[ -]?rated\b|\bpoor(?:ly)? rated\b|\bbad rating\b|\blow customer rating\b',
 'average':r'\b(?:3|three) (?:out of|of) (?:5|five)\b|\b(?:3|three)[ -]?star\b|\baverage (?:customer )?rating\b|\bmoderate(?:ly)? rated\b|\bmedium rating\b|\baverage rated\b',
 'high':r'\b(?:5|five) (?:out of|of) (?:5|five)\b|\b(?:5|five)[ -]?star\b|\bhigh(?:ly)?[ -]?rated\b|\bhigh customer rating\b|\bexcellent rating\b|\btop rating\b|\bgreat ratings\b'
},
'area':{
 'riverside':r'\briverside\b|\bby the river\b|\bnear (?:the )?river\b|\bon the river\b|\balong the river\b',
 'city':r'\bcity centre\b|\bcity center\b|\bcentre of (?:the )?city\b'
}
}

CATS={k:{l:re.compile(p) for l,p in d.items()} for k,d in CATS.items()}

def slots(t):
 s=norm(t);return {group:{label for label,pat in defs.items() if pat.search(s)} for group,defs in CATS.items()}

NAMELIST=['Alimentum','Aromi','Bibimbap House','Blue Spice','Browns Cambridge','Clowns','Cocum','Cotto','Fitzbillies','Giraffe','Green Man','Loch Fyne','Midsummer House','Strada','The Cambridge Blue','The Cricketers','The Eagle','The Golden Palace','The Mill','The Olive Grove','The Phoenix','The Plough','The Punter','The Rice Boat','The Vaults','The Waterman','The Wrestlers','The Twenty Two','Travellers Rest Beefeater','Wildwood','Zizzi','The Twenty Two','The Rice Boat','The Portland Arms','The Sorrento','Café Sicilia','Café Brazil','Café Rouge','Café Adriatic','Rainbow Vegetarian Café','Yippee Noodle Bar','Crowne Plaza Hotel','The Bakers','Raja Indian Cuisine','All Bar One','Clare Hall','Avalon','The Olive Grove','The Cricketers','The Cambridge Blue','The Plough']
LANDMARKS=['All Bar One','Avalon','Café Adriatic','Café Brazil','Café Rouge','Café Sicilia','Clare Hall','Crowne Plaza Hotel','Raja Indian Cuisine','Rainbow Vegetarian Café','The Bakers','The Portland Arms','The Sorrento','Yippee Noodle Bar','The Rice Boat','The Olive Grove']

# Ground truth is majority references (2+ or plurality) for each slot. Used only within same card.
truth={}
for key,refs in refmap.items():
 vals=[slots(r) for r in refs]; d={}
 for typ in CATS:
  counts=collections.Counter([val for v in vals for val in v[typ]])
  d[typ]={val for val,n in counts.items() if n>=2}
 d['names']={norm(n).replace('the ','',1) for n in NAMELIST if sum(norm(n).replace('the ','',1) in norm(r).replace('the ','') for r in refs)>=3 and n not in LANDMARKS}
 d['landmarks']={norm(n) for n in LANDMARKS if sum(norm(n) in norm(r) for r in refs)>=2}
 truth[key]=d

def features(t,key):
 expected=truth[key]; seen=slots(t); s=norm(t)
 rows={};
 for typ,refv in expected.items():
  if typ in CATS:
   got=seen[typ];rows[typ+'_covered']=len(got&refv);rows[typ+'_missing']=len(refv-got);rows[typ+'_wrong']=len(got-refv)
  elif typ=='names':
   # exclude near places that are themselves cities in other cards
   if refv:
    rows['name_missing']=int(not any(n in s.replace('the ','') for n in refv))
   else:rows['name_missing']=0
  elif typ=='landmarks':rows['landmark_missing']=int(bool(refv) and not any(n in s for n in refv))
 return rows

F=[]
for r in P.itertuples():
 f=features(r.text,(r.round,r.card));f.update(round_=r.round,seat=r.seat,card=r.card,text=r.text);F.append(f)
f=pd.DataFrame(F)
agg=f.drop(columns=['card','text']).groupby(['round_','seat']).sum().round(2)
if __name__=='__main__':
 print(agg.to_string())
 f.to_csv(HERE / 'semantic_per_card.csv',index=False)
 agg.to_csv(HERE / 'semantic_per_deck.csv')
