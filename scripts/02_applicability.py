import pandas as pd
import os
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem import Descriptors
import joblib
from tqdm import tqdm

root = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(os.path.join(root, "..", "data", "cemm_smiles.csv"))
dataset_smiles = df["smiles"].tolist()

dataset_mols = [Chem.MolFromSmiles(smiles) for smiles in dataset_smiles]

dataset_fps = [
    AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) for mol in dataset_mols
]

joblib.dump(dataset_fps, os.path.join(root, "..", "models", "cemm_ecfp_2_1024.joblib"))

ds = pd.read_csv(os.path.join(root, "..", "data", "all_fff_enamine.csv"))

all_query_smiles = ds["smiles"].tolist()

sims_1 = []
sims_3 = []
logps = []
mwts = []
for query_smiles in tqdm(all_query_smiles):
    query_mol = Chem.MolFromSmiles(query_smiles)
    query_fp = AllChem.GetMorganFingerprintAsBitVect(query_mol, 2, nBits=1024)
    similarity_scores = [
        DataStructs.TanimotoSimilarity(query_fp, dataset_fp)
        for dataset_fp in dataset_fps
    ]
    sorted_scores_indices = sorted(
        enumerate(similarity_scores), key=lambda x: x[1], reverse=True
    )
    top_n = 3
    sims_1 += [sorted_scores_indices[0][1]]
    sims_3 += [sorted_scores_indices[2][1]]
    logps += [Descriptors.MolLogP(query_mol)]
    mwts += [Descriptors.MolWt(query_mol)]

df = pd.DataFrame({"sims-1": sims_1, "sims-3": sims_3, "logp": logps, "mw": mwts})
df.to_csv(os.path.join(root, "..", "data", "applicability.csv"), index=False)
