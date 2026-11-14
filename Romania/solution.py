"""Shaken, Still Standing - mixture-of-experts solution.

Idea: the 7500 + 5000 records come from THREE different networks, and each network
uses a DIFFERENT rule to decide "Vrancea or not". One single model cannot learn three
rules at once (logistic regression ~0.60, SVM ~0.68, gradient boosting ~0.85), so we

  1. find out which network a record comes from (the "gate"),
  2. train one small expert per network,
  3. let each record be decided by the expert of its own network.

Run from the folder that contains dataset/ and evaluation_script.py:
    python solution.py
"""
import subprocess, sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# ------------------------------------------------------------------ 1. load
train = pd.read_csv("dataset/train.csv")
sealed = pd.read_csv("dataset/sealed.csv")
ALL = [f"c{j + 1:02d}" for j in range(48)]
y = train.vrancea.to_numpy()

# Network C only uses these 12 channels; its other 36 channels are weak noise (x0.25 amplitude).
CHANNELS_C = ["c03", "c09", "c12", "c14", "c15", "c19", "c27", "c28", "c30", "c36", "c39", "c40"]
NOISY_C = [c for c in ALL if c not in CHANNELS_C]

# ------------------------------------------------------------------ 2. the gate
# (a) network C: its 36 "noisy" channels are 4x smaller than in the other networks.
both = pd.concat([train, sealed])
mean, std = both[NOISY_C].mean(), both[NOISY_C].std()
for _ in range(2):                      # 2 passes: first rough, then without the small rows
    def spread(df):
        return (((df[NOISY_C] - mean) / std) ** 2).mean(axis=1).to_numpy()
    keep = spread(both) > 0.3
    mean, std = both.loc[keep.tolist(), NOISY_C].mean(), both.loc[keep.tolist(), NOISY_C].std()

def network(df):
    """0 = A, 1 = B, 2 = C"""
    net = np.where(df.c13.to_numpy() < -0.6, 0, 1)   # (b) A / B are split by channel c13 at -0.6
    net[spread(df) < 0.25] = 2
    return net

net_train, net_sealed = network(train), network(sealed)
print("records per network  train:", np.bincount(net_train), " sealed:", np.bincount(net_sealed))

# ------------------------------------------------------------------ 3. the three experts
# Expert A: parity (XOR) of three thresholds. Thresholds were found with a grid search.
T43, T35, T38 = -0.61, -0.23, 0.73
def expert_A(df):
    above = (df.c43 > T43).astype(int) + (df.c35 > T35).astype(int) + (df.c38 > T38).astype(int)
    return 1 - above % 2

# Expert B: plain logistic regression on all channels except c13 (the gate channel).
B_CHANNELS = [c for c in ALL if c != "c13"]
make_B = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=5000))

# Expert C: logistic regression on squares and products of the 12 channels (an ellipsoid).
make_C = lambda: make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False),
                               LogisticRegression(C=0.1, max_iter=10000))

# ------------------------------------------------------------------ 4. scores (5-fold CV on train)
cv = StratifiedKFold(5, shuffle=True, random_state=0)
a, b, c = net_train == 0, net_train == 1, net_train == 2
print("\nexpert A  parity of 3 thresholds :", round((expert_A(train[a]) == y[a]).mean(), 4), "(3 fitted numbers only)")
print("expert B  logistic regression    :", round(cross_val_score(make_B(), train.loc[b, B_CHANNELS], y[b], cv=cv).mean(), 4))
print("expert C  quadratic logistic reg.:", round(cross_val_score(make_C(), train.loc[c, CHANNELS_C], y[c], cv=cv).mean(), 4))

# ------------------------------------------------------------------ 5. predict the sealed records
pred = np.zeros(len(sealed), dtype=int)
A, B, C = net_sealed == 0, net_sealed == 1, net_sealed == 2
pred[A] = expert_A(sealed[A])
pred[B] = make_B().fit(train.loc[b, B_CHANNELS], y[b]).predict(sealed.loc[B, B_CHANNELS])
pred[C] = make_C().fit(train.loc[c, CHANNELS_C], y[c]).predict(sealed.loc[C, CHANNELS_C])

pd.DataFrame({"record_id": sealed.record_id, "vrancea": pred}).to_csv("predictions.csv", index=False)
print("\nwrote predictions.csv:", np.bincount(pred))

# ------------------------------------------------------------------ 6. ask the evaluator
print("\nevaluator says:")
sys.stdout.flush()
subprocess.run([sys.executable, "evaluation_script.py", "predictions.csv"])
