"""
Camera-ready analyses for the DEEP models (TabNet, FT-Transformer).
Same training code/settings as benchmark.ipynb (cells 10-12, 17-18).
Produces, for each model:
  - per-fold acc/F1/sens/spec for leaky + honest 5-fold CV
  - paper's 80/20 temporal split: acc/F1/sens/spec + binomial vs test majority
  - expanding-window temporal (train first c, test next 10%, c in .6/.7/.8/.9)

Usage:  python analysis_deep.py thesis_clean.csv results_cr  [--only tabnet|ftt] [--parts cv,temporal,expanding]
(fold column = CV fold for leaky/honest, = seed for temporal/expanding)
Writes rows incrementally to results_cr/deep_rows.csv (safe to resume: finished jobs are skipped).
"""
import sys, os, argparse, warnings, time
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from scipy import stats
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("csv"); ap.add_argument("out")
ap.add_argument("--only", default="tabnet,ftt")
ap.add_argument("--parts", default="cv,temporal,expanding")
ap.add_argument("--epochs", type=int, default=100)
ap.add_argument("--seeds", default="42,0,1", help="seeds for temporal/expanding runs (42 = paper)")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
ROWS = f"{args.out}/deep_rows.csv"

from pytorch_tabnet.tab_model import TabNetClassifier
from rtdl_revisiting_models import FTTransformer

df = pd.read_csv(args.csv)
fc = [c for c in df.columns if c not in ["Date", "Quality", "signature"]]
de = df.copy()
for c in fc:
    if not pd.api.types.is_numeric_dtype(de[c]):
        de[c] = LabelEncoder().fit_transform(de[c].astype(str))
le = LabelEncoder().fit(df["Quality"].astype(str)); de["Quality"] = le.transform(df["Quality"].astype(str))
BAD = int(le.transform(["Bad"])[0])
X_np = de[fc].values.astype("float32"); y_np = de["Quality"].values.astype("int64")
groups_np = df["signature"].values
cat_cols = [c for c in fc if not pd.api.types.is_numeric_dtype(df[c])]
cat_idxs = [fc.index(c) for c in cat_cols]
cat_dims = [int(de[c].nunique()) for c in cat_cols]
cont_idxs = [i for i in range(len(fc)) if i not in cat_idxs]
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, flush=True)

def fit_tabnet(Xtr, ytr, Xte, seed=42):
    clf = TabNetClassifier(cat_idxs=cat_idxs, cat_dims=cat_dims, cat_emb_dim=1,
                           seed=seed, verbose=0, device_name="auto")
    clf.fit(Xtr, ytr, max_epochs=args.epochs, patience=0, batch_size=256, virtual_batch_size=64)
    return clf.predict(Xte)

def fit_ftt(Xtr, ytr, Xte, seed=42, lr=1e-3, batch_size=256):
    sc = StandardScaler().fit(Xtr[:, cont_idxs])
    tc = lambda A: torch.tensor(sc.transform(A[:, cont_idxs]), dtype=torch.float32, device=device)
    tk = lambda A: torch.tensor(A[:, cat_idxs], dtype=torch.long, device=device)
    Xtr_c, Xte_c, Xtr_k, Xte_k = tc(Xtr), tc(Xte), tk(Xtr), tk(Xte)
    yt = torch.tensor(ytr, dtype=torch.long, device=device)
    m = FTTransformer(n_cont_features=len(cont_idxs), cat_cardinalities=cat_dims,
                      d_out=2, **FTTransformer.get_default_kwargs()).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-5)
    n = Xtr_c.shape[0]; m.train()
    for _ in range(args.epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            b = perm[i:i + batch_size]; opt.zero_grad()
            F.cross_entropy(m(Xtr_c[b], Xtr_k[b]), yt[b]).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        return m(Xte_c, Xte_k).argmax(1).cpu().numpy()

FITTERS = {"tabnet": ("TabNet", fit_tabnet), "ftt": ("FT-Transformer", fit_ftt)}

# ---- job list ----
N = len(df)
dft = df.copy(); dft["dt"] = pd.to_datetime(dft["Date"], format="%m/%d/%Y", errors="coerce")
paper_order = dft.sort_values("dt").index.to_numpy()      # identical to notebook cells 17/18
SEEDS = [int(x) for x in args.seeds.split(",")]
jobs = []
if "cv" in args.parts:
    for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(X_np, y_np)):
        jobs.append(("leaky", k, tr, te, 42))
    for k, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(X_np, y_np, groups_np)):
        jobs.append(("honest", k, tr, te, 42))
if "temporal" in args.parts:
    c = int(N * .8)
    for sd in SEEDS: jobs.append(("temporal8020", sd, paper_order[:c], paper_order[c:], sd))
if "expanding" in args.parts:
    for c in [.6, .7, .8, .9]:
        a = int(N * c); b = int(N * (c + .1)) if c < .9 else N
        for sd in SEEDS: jobs.append((f"expanding{c}", sd, paper_order[:a], paper_order[a:b], sd))

done = set()
if os.path.exists(ROWS):
    d = pd.read_csv(ROWS); done = set(zip(d.model, d.protocol, d.fold))

for key in args.only.split(","):
    name, fit = FITTERS[key]
    for proto, k, tr, te, sd in jobs:
        if (name, proto, k) in done: continue
        torch.manual_seed(sd); np.random.seed(sd)
        t0 = time.time()
        p = fit(X_np[tr], y_np[tr], X_np[te], seed=sd); yt = y_np[te]
        maj = max((yt == BAD).mean(), 1 - (yt == BAD).mean())
        row = dict(model=name, protocol=proto, fold=k, seed=sd, n=len(yt),
                   acc=accuracy_score(yt, p), f1=f1_score(yt, p, average="macro"),
                   sens=(p[yt == BAD] == BAD).mean(), spec=(p[yt != BAD] != BAD).mean(),
                   majority=maj, binom_p=stats.binomtest(int((p == yt).sum()), len(yt), maj).pvalue,
                   pred_bad_frac=(p == BAD).mean(), secs=round(time.time() - t0, 1), device=device)
        pd.DataFrame([row]).to_csv(ROWS, mode="a", header=not os.path.exists(ROWS), index=False)
        print(f"{name:15s} {proto:13s} fold {k}  acc {row['acc']:.4f}  f1 {row['f1']:.4f}  "
              f"sens {row['sens']:.3f} spec {row['spec']:.3f}  ({row['secs']}s)", flush=True)
print("ALL DONE")
