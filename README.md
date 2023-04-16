# Fragment predictor app
A small prediction app for fully functionalized fragments

## Installation

```bash
conda create -n fpred python=3.10
conda activate fpred

# download and install fragment-embedding
git clone https://github.com/ligand-discovery/fragment-embedding.git
cd fragment-embedding
python -m pip install -e .
cd ..

# download and install mini-automl
git clone https://github.com/ligand-discovery/mini-automl.git
cd mini-automl
python -m pip install -e .
cd ..

# downloa and install the current repository
git clone https://github.com/ligand-discovery/fragment-predictor.git
cd fragment-predictor
python -m pip install -r requirements.txt
```

## Run the app

```bash
streamlit run app/app.py
```