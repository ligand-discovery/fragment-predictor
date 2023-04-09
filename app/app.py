import streamlit as st
from fragmentembedding import FragmentEmbedder
from rdkit import Chem
import pandas as pd
import os

root = os.path.dirname(os.path.abspath(__file__))

fe = FragmentEmbedder()

enamine_catalog = pd.read_csv(os.path.join(root, "..", "data", "all_fff_enamine.csv"))
enamine_catalog_ids_set = set(enamine_catalog["catalog_id"])

def is_enamine_catalog_id(identifier):
    if identifier in enamine_catalog_ids_set:
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

def has_crf(smiles):
    pattern = "CC1(CCC#C)N=N1"


st.title("Fully-functionalized fragment predictions")

text_input = st.text_area(label="Fragments input")

st.write(text_input)

st.sidebar.title("Models")
