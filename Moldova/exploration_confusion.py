import pandas as pd
from collections import Counter

o = pd.read_csv('observations.csv')
G = ['ARKX', 'BSGZ', 'CDOQ', 'EFPL', 'HNWM', 'IYJT']      # litere asemănătoare
g = {c: i for i, s in enumerate(G) for c in s}
compat = lambda a, b: all(p == '?' or g[p] == g[q] for p, q in zip(a, b))

reg = pd.read_csv('ring_register.csv')
reads = o[(o.species == 'Ciconia ciconia') & o.ring_read.notna()]
storks = reg[reg.species == 'Ciconia ciconia'].ring

# citiri care nu se potrivesc cu nicio berză din registru
unk = reads[~reads.ring_read.map(lambda x: any(compat(x, k) for k in storks))]
print(unk.locality_id.unique())                            # -> L5819951 (Beleu)

# vot majoritar pe poziții = inelul "văzut"
seen = ''.join(Counter(x[i] for x in unk.ring_read if x[i] != '?').most_common(1)[0][0]
               for i in range(6))                          # RHGLWQ

# permutarea de confuzie (tipul P) inversată, deoarece e o involuție
P = dict(zip('AKRXBGSZCODQEFLPHWMNIYJT', 'KAXRGBZSOCQDFEPLWHNMYITJ'))
print('SIGMOID_' + ''.join(P[c] for c in seen))            # SIGMOID_XWBPHD
