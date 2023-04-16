import os
import joblib
import numpy as np
from tqdm import tqdm
import pandas as pd
from rdkit import Chem
from tqdm import tqdm
from fragmentembedding import FragmentEmbedder

root = os.path.dirname(os.path.abspath(__file__))

dm = pd.read_csv(os.path.join(root, "..", "data", "model_catalog.csv"))

models = {}
all_models = dm["model_name"].tolist()
for r in all_models:
    models[r] = joblib.load(os.path.join(root, "..", "models", "{0}.joblib".format(r)))

smiles_list = pd.read_csv(os.path.join(root, "..", "data", "all_fff_enamine.csv"))[
    "smiles"
].tolist()
inchikeys = [
    Chem.MolToInchiKey(Chem.MolFromSmiles(smiles)) for smiles in tqdm(smiles_list)
]

fe = FragmentEmbedder()


def chunk_list(input_list, chunk_size):
    return [
        input_list[i : i + chunk_size] for i in range(0, len(input_list), chunk_size)
    ]


idxs = [i for i in range(len(inchikeys))]

chunks_idxs = []
chunk_idx = 0
for idxs_ in chunk_list(idxs, 10000):
    smiles_sublist = [smiles_list[i] for i in idxs_]
    inchikeys_sublist = [inchikeys[i] for i in idxs_]
    X = fe.transform(smiles_sublist)
    R = []
    for model_name in tqdm(all_models):
        model = models[model_name]
        y_hats = model.predict(X)
        R += [y_hats]
    R = np.array(R).T
    dr = pd.DataFrame(R, columns=all_models)
    dr["inchikey"] = inchikeys_sublist
    dr = dr[["inchikey"] + all_models]
    dr.to_csv(
        os.path.join(root, "..", "data", "predictions_cache_{0}.csv".format(chunk_idx)),
        index=False,
    )
    chunks_idxs += [chunk_idx]
    chunk_idx += 1
