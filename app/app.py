import streamlit as st
from fragmentembedding import FragmentEmbedder
from rdkit import Chem
import pandas as pd
import os
import random
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem import Descriptors
from scipy import stats
import textwrap

import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Fragment predictor app",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon=None,
)

root = os.path.dirname(os.path.abspath(__file__))

df_predictions = pd.read_csv(os.path.join(root, "..", "data", "predictions.csv"))
predictions_inchikeys = df_predictions["inchikey"].tolist()
df_predictions = df_predictions.rename(columns={"inchikey": "InChIKey"})
df_applicability = pd.read_csv(os.path.join(root, "..", "data", "applicability.csv"))
df_predictions = pd.concat([df_predictions, df_applicability], axis=1)

fid2smi = {}
for r in pd.read_csv(os.path.join(root, "..", "data", "cemm_smiles.csv")).values:
    fid2smi[r[0]] = r[1]

fe = FragmentEmbedder()

CRF_PATTERN = "CC1(CCC#C)N=N1"
CRF_PATTERN_0 = "C#CC"
CRF_PATTERN_1 = "N=N"

enamine_catalog = pd.read_csv(os.path.join(root, "..", "data", "all_fff_enamine.csv"))
enamine_catalog_ids_set = set(enamine_catalog["catalog_id"])
enamine_catalog_dict = {}
catalog2inchikey = {}
smiles2catalog = {}
for i, r in enumerate(enamine_catalog.values):
    enamine_catalog_dict[r[0]] = r[1]
    catalog2inchikey[r[0]] = predictions_inchikeys[i]
    smiles2catalog[r[1]] = r[0]


def is_enamine_catalog_id(identifier):
    if identifier in enamine_catalog_ids_set:
        return True
    else:
        return False


def is_enamine_smiles(identifier):
    if identifier in smiles2catalog:
        return True
    else:
        return False


def is_ligand_discovery_id(identifier):
    if identifier in fid2smi:
        return True
    else:
        return False


def is_valid_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
    except:
        mol = None
    if mol is None:
        return False
    else:
        return True


def has_crf(mol):
    pattern = CRF_PATTERN
    has_pattern = mol.HasSubstructMatch(Chem.MolFromSmarts(pattern))
    if not has_pattern:
        if mol.HasSubstructMatch(
            Chem.MolFromSmarts(CRF_PATTERN_0)
        ) and mol.HasSubstructMatch(Chem.MolFromSmarts(CRF_PATTERN_1)):
            return True
        else:
            return False
    return True


st.title("Fully-functionalized fragment predictions")

dm = pd.read_csv(os.path.join(root, "..", "data", "model_catalog.csv"))
dp = pd.read_csv(os.path.join(root, "..", "data", "models_performance.tsv"), sep="\t")

model_display = {}
model_description = {}
for r in dm.values:
    model_display[r[0]] = r[1]
    model_description[r[0]] = r[2]
model_auroc = {}
for r in dp.values:
    model_auroc[r[0]] = r[1]

prom_models = [x for x in dm["model_name"].tolist() if x.startswith("promiscuity")]
sign_models = [x for x in dm["model_name"].tolist() if x.startswith("signature")]


def model_to_markdown(model_names):
    items = []
    for mn in model_names:
        items += [
            "{0} ({1:.3f}): {2}".format(
                model_display[mn].ljust(8), model_auroc[mn], model_description[mn]
            )
        ]
    markdown_list = "\n".join(items)
    return markdown_list


st.sidebar.title("Promiscuity models")

st.sidebar.markdown("**Global models**")

global_promiscuity_models = ["promiscuity_pxf0", "promiscuity_pxf1", "promiscuity_pxf2"]
st.sidebar.text(model_to_markdown(global_promiscuity_models))

st.sidebar.markdown("**Specific models**")

specific_promiscuity_models = [
    "promiscuity_fxp0_pxf0",
    "promiscuity_fxp1_pxf0",
    "promiscuity_fxp2_pxf0",
    "promiscuity_fxp0_pxf1",
    "promiscuity_fxp1_pxf1",
    "promiscuity_fxp2_pxf1",
    "promiscuity_fxp0_pxf2",
    "promiscuity_fxp1_pxf2",
    "promiscuity_fxp2_pxf2",
]
st.sidebar.text(model_to_markdown(specific_promiscuity_models))

st.sidebar.markdown("**Aggregated score**")
st.sidebar.text("Sum             : Sum of individual promiscuity predictors.")

st.sidebar.title("Signature models")
signature_models = ["signature_{0}".format(i) for i in range(10)]
st.sidebar.text(model_to_markdown(signature_models))

st.sidebar.title("Chemical space")
s = ["MW              : Molecular weight.",
     "LogP            : Walden-Crippen LogP.",
     "Sim-1           : Tanimoto similarity to the most ",
     "                  similar fragment in the training set.",
     "Sim-3           : Tanimoto similarity to the third ",
     "                  most similar fragment in the training set."]

st.sidebar.text("\n".join(s))

s = textwrap.wrap("*  The score in parenthesis corresponds to the mean AUROC in 10 train-test splits.")
st.sidebar.text("\n".join(s))

st.sidebar.markdown("**In the main page...**")
s = textwrap.wrap("1. Percentages in parenthesis denote the percentile of the score across the Enamine collection of FFFs (>250k compounds)", width=60)
st.sidebar.text("\n".join(s))
s = textwrap.wrap("2. The exclamation sign (!) indicates that the corresponding model has an AUROC accuracy below 0.7.", width=60)
st.sidebar.text("\n".join(s))


models = {}
all_models = dm["model_name"].tolist()
for r in all_models:
    models[r] = os.path.join(root, "..", "models", "{0}.joblib".format(r))

placeholder_text = []
keys = random.sample(enamine_catalog_ids_set, 5)
for k in keys:
    placeholder_text += [random.choice([k, enamine_catalog_dict[k]])]
placeholder_text = "\n".join(placeholder_text)

text_input = st.text_area(label="Input your fully functionalized fragments:")
inputs = [x.strip(" ") for x in text_input.split("\n")]
inputs = [x for x in inputs if x != ""]
if len(inputs) > 999:
    st.error("Please limit the number of input fragments to 999.")

R = []
all_inputs_are_valid = True
for i, inp in enumerate(inputs):
    input_id = "input-{0}".format(str(i).zfill(3))
    if is_enamine_catalog_id(inp):
        smiles = enamine_catalog_dict[inp]
        inchikey = catalog2inchikey[inp]
        r = [inp, smiles, inchikey]
    elif is_ligand_discovery_id(inp):
        smiles = fid2smi[inp]
        inchikey = Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))
        r = [inp, smiles, inchikey]
    elif is_enamine_smiles(inp):
        smiles = inp
        inp = smiles2catalog[smiles]
        inchikey = catalog2inchikey[inp]
        r = [inp, smiles, inchikey]
    elif is_valid_smiles(inp):
        mol = Chem.MolFromSmiles(inp)
        if has_crf(mol):
            inchikey = Chem.rdinchi.InchiToInchiKey(Chem.MolToInchi(mol))
            r = [inchikey, inp, inchikey]
        else:
            st.error(
                "Input SMILES {0} does not have the CRF. The CRF pattern is {1}.".format(
                    inp, CRF_PATTERN
                )
            )
            all_inputs_are_valid = False
    else:
        st.error(
            "Input {0} is not valid. Please enter a valid fully-functionalized fragment SMILES string or an Enamine catalog identifier of a fully-functionalized fragment".format(
                inp
            )
        )
        all_inputs_are_valid = False
    R += [r]


def get_fragment_image(smiles):
    m = Chem.MolFromSmiles(smiles)
    AllChem.Compute2DCoords(m)
    opts = Draw.DrawingOptions()
    opts.bgColor = None
    im = Draw.MolToImage(m, size=(200, 200), options=opts)
    return im


if all_inputs_are_valid and len(R) > 0:
    sum_of_promiscuities = np.sum(
        df_predictions[global_promiscuity_models + specific_promiscuity_models], axis=1
    )
    df = pd.DataFrame(R, columns=["Identifier", "SMILES", "InChIKey"])

    my_inchikeys = df["InChIKey"].tolist()

    df_done = df[df["InChIKey"].isin(predictions_inchikeys)]
    df_todo = df[~df["InChIKey"].isin(predictions_inchikeys)]

    if df_done.shape[0] > 0:
        df_done = df_done.merge(
            df_predictions, on="InChIKey", how="left"
        ).drop_duplicates()

    if df_todo.shape[0] > 0:
        X = fe.transform(df_todo["SMILES"].tolist())

        st.info("Making predictions... this make take a few seconds. Please be patient. We may experience high traffic. If something goes wrong, please try again later.")

        progress_bar = st.progress(0)

        for i, model_name in enumerate(all_models):
            print(model_name)
            model = joblib.load(models[model_name])
            vals = model.predict(X)
            del model
            progress_bar.progress((i + 1) / len(all_models))
            df_todo[model_name] = vals

        dataset_fps = joblib.load(
            os.path.join(root, "..", "models", "cemm_ecfp_2_1024.joblib")
        )

        all_query_smiles = df_todo["SMILES"].tolist()

        sims_1 = []
        sims_3 = []
        logps = []
        mwts = []
        for query_smiles in all_query_smiles:
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
        results = {"sims-1": sims_1, "sims-3": sims_3, "logp": logps, "mw": mwts}
        for k in ["sims-1", "sims-3", "logp", "mw"]:
            df_todo[k] = results[k]

    if df_done.shape[0] > 0 and df_todo.shape[0] > 0:
        df_ = pd.concat([df_done, df_todo])
    else:
        if df_done.shape[0] > 0:
            df_ = df_done
        else:
            df_ = df_todo
    df_ = df_.drop(columns=["Identifier", "SMILES"])
    df = df.merge(df_, on="InChIKey", how="left")
    df.drop_duplicates(subset=['InChIKey'], keep='first', inplace=True, ignore_index=True)
    df = df.rename(columns=model_display)
    applicability_display = {
        "mw": "MW",
        "logp": "LogP",
        "sims-1": "Sim-1",
        "sims-3": "Sim-3",
    }
    df = df.rename(columns=applicability_display)

    df_predictions = df_predictions.rename(columns=model_display)
    df_predictions = df_predictions.rename(columns=applicability_display)

    prom_columns = []
    for i in range(3):
        prom_columns += ["Prom-{0}".format(i)]
        for j in range(3):
            prom_columns += ["Prom-{0}-{0}".format(i, j)]

    def identifiers_text(ik, smi, ident):
        s = ["{0}".format(ik), "{0}".format(smi)]
        if ik != ident:
            s += ["{0}".format(ident)]
        return "\n".join(s)

    def score_text(v, c):
        all_scores = np.array(df_predictions[c])
        perc = stats.percentileofscore(all_scores, v)
        t = "{0}: {1:.2f} ({2:.1f}%)".format(c.ljust(8), v, perc).ljust(22)
        if c == "Sign-4" or c == "Sign-7" or c == "Sign-3":
            t += " (!)"
        return t

    def score_texts(vs, cs):
        all_texts = []
        for v, c in zip(vs, cs):
            all_texts += [score_text(v, c)]
        return "\n".join(all_texts)

    dorig = pd.DataFrame({"InChIKey": my_inchikeys})
    df = dorig.merge(df, on="InChIKey", how="left")
    df = df.reset_index(inplace=False, drop=True)
    for i, r in enumerate(df.iterrows()):
        v = r[1]
        st.markdown("#### Input {0}: `{1}`".format(i+1, inputs[r[0]]))
        cols = st.columns(4)
        cols[0].markdown("**Fragment**")
        cols[0].image(get_fragment_image(v["SMILES"]))
        cols[0].text(identifiers_text(v["InChIKey"], v["SMILES"], v["Identifier"]))

        cols[1].markdown("**Chemical space**")
        my_cols = ["MW", "LogP", "Sim-1", "Sim-3"]
        cols[1].text(score_texts(v[my_cols], my_cols))

        cols[2].markdown("**Promiscuity**")
        sum_prom = np.sum(v[prom_columns])
        perc_prom = stats.percentileofscore(sum_of_promiscuities, sum_prom)
        cols[2].text("Sum     : {0:.2f} ({1:.1f}%)".format(sum_prom, perc_prom))
        my_cols = ["Prom-0", "Prom-1", "Prom-2"]
        cols[2].text(score_texts(v[my_cols], my_cols))

        my_cols = [
            "Prom-0-0",
            "Prom-0-1",
            "Prom-0-2",
            "Prom-1-0",
            "Prom-1-1",
            "Prom-1-2",
            "Prom-2-0",
            "Prom-2-1",
            "Prom-2-2",
        ]
        cols[2].text(score_texts(v[my_cols], my_cols))

        cols[3].markdown("**Signatures**")
        my_cols = ["Sign-{0}".format(i) for i in range(10)]
        cols[3].text(score_texts(v[my_cols], my_cols))

    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False).encode("utf-8")

    csv = convert_df(df)

    st.download_button(
        "Download as CSV", csv, "predictions.csv", "text/csv", key="download-csv"
    )

else:
    st.info(
        "This tool expects fully functionalized fragments (FFF) as input, including the diazirine+alkyne probe (CRF). We have tailored the chemical space of the predictions to FFFs; the app will through an error if any of the input molecules does not contain a CRF region. Enamine provides a good [catalog](https://enamine.net/compound-libraries/fragment-libraries/fully-functionalized-probe-library) of FFFs. For a quick test input, use any of the options below."
    )

    example_0 = ["Z5645472552", "Z5645472643", "Z5645472785"]
    st.markdown("**Input Enamine FFF identifiers...**")
    st.text("\n".join(example_0))

    example_1 = [
        "C#CCCC1(CCCNC(=O)C(Cc2c[nH]c3ncccc23)NC(=O)OC(C)(C)C)N=N1",
        "C#CCCC1(CCCNC(=O)[C@H]2CCC(=O)NC2)N=N1",
        "C#CCCC1(CCCNC(=O)CSc2ncc(C(=O)OCC)c(N)n2)N=N1",
    ]
    st.markdown("**Input FFF SMILES strings...**")
    st.text("\n".join(example_1))

    example_2 = ["C310", "C045", "C391"]
    st.markdown("**Input Ligand Discovery identifiers...**")
    st.text("\n".join(example_2))

    example_3 = [
        "Z5645486561",
        "C#CCCCC1(CCCC(=O)N2CCC(C(C(=O)O)c3ccc(C)cc3)CC2)N=N1",
        "C279",
    ]
    st.markdown("**Input a mix of the above identifiers**")
    st.text("\n".join(example_3))
