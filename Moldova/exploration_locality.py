import pandas as pd
from collections import Counter
import numpy as np

r = pd.read_csv('ring_register.csv')
o = pd.read_csv('observations.csv')

# citiri de berze albe cu inel
s = o[(o.species == 'Ciconia ciconia') & o.ring_read.notna()]
loc = o.drop_duplicates('locality_id').set_index('locality_id')[['latitude', 'longitude']]
storks = r[r.species == 'Ciconia ciconia'].reset_index(drop=True)

# distanța (km) de la fiecare localitate la cea mai apropiată berză din registru
for L, n in s.groupby('locality_id').size().sort_values(ascending=False).head(14).items():
    la, lo = loc.loc[L]
    d = np.hypot((storks.ringed_latitude - la) * 111,
                 (storks.ringed_longitude - lo) * 77)
    print(L, n, round(d.min(), 2), storks.ring[d.idxmin()])
# L5819951 iese în evidență: 24 citiri, cea mai apropiată berză din registru la 74 km

# vot majoritar pe poziții, fără '?'
reads = s[s.locality_id == 'L5819951'].ring_read
ring = ''.join(Counter(x[k] for x in reads if x[k] != '?').most_common(1)[0][0]
               for k in range(6))
print('SIGMOID_' + ring)   # SIGMOID_RHGLWQ
