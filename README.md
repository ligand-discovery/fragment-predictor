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

# download and install tabpfn with cpu requirements
pip install torch --index-url https://download.pytorch.org/whl/cpu
git clone https://github.com/DhanshreeA/TabPFN.git
pip install TabPFN/.
rm -rf TabPFN

# download and install the current repository
git clone https://github.com/ligand-discovery/fragment-predictor.git
cd fragment-predictor
python -m pip install -r requirements.txt
```

## Download the necessary data

In the `fragment-predictor/` folder root, download and unzip the following folders:
* `models`: https://ligand-discovery.s3.eu-central-1.amazonaws.com/fragment-predictor/models.zip
* `data`: https://ligand-discovery.s3.eu-central-1.amazonaws.com/fragment-predictor/data.zip


## Run the app

```bash
streamlit run app/app.py
```