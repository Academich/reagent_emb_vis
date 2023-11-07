import math
import json
from collections import Counter
from itertools import combinations
import pandas as pd
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds
from umap import UMAP

PATH = 'benchmark_uspto/tgt-train-1181413.txt'
REAGENTS_PATH = "benchmark_uspto/reagents.json"
SEPARATOR = ";"


target = pd.read_csv(PATH, header=None)[0].str.split(SEPARATOR)
with open(REAGENTS_PATH) as f:
    i2x = {int(k): v for k, v in json.load(f).items()}
    x2i = {v: k for k, v in i2x.items()}

cx = Counter()
cxy = Counter()
for entry in target:
    for reagent in entry:
        cx[reagent] += 1
    for reagent_1, reagent_2 in map(sorted, combinations(entry, 2)):
        cxy[(reagent_1, reagent_2)] += 1

sx = sum(cx.values())
sxy = sum(cxy.values())

pmi_samples = {}
data, rows, cols = [], [], []
for (x, y), n in cxy.items():
    rows.append(x2i[x])
    cols.append(x2i[y])
    data.append(math.log((n / sxy) / (cx[x] / sx) / (cx[y] / sx)))
    pmi_samples[(x, y)] = data[-1]
PMI = csc_matrix((data, (rows, cols)))

U, _, _ = svds(PMI, k=50)
norms = np.sqrt(np.sum(np.square(U), axis=1, keepdims=True))
U /= np.maximum(norms, 1e-7)

xy = UMAP().fit_transform(U)
sphere_mapper = UMAP(output_metric='haversine', random_state=12345, n_neighbors=30).fit(U)
theta = sphere_mapper.embedding_[:, 0]
phi = sphere_mapper.embedding_[:, 1]
smiles = [None] * len(i2x)
for i in i2x:
    smiles[i] = i2x[i]
rgs_benchmark_umap_50d = pd.concat(
    (pd.DataFrame(np.hstack((xy, sphere_mapper.embedding_))),
     pd.Series(smiles)), axis=1
)
rgs_benchmark_umap_50d.columns = ["x", "y", "theta", "phi", "smiles"]

# TODO roles from saved dict
# TODO names from saved dict


