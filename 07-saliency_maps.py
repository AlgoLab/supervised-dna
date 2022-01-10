"""
Analyze the most influencial kmers in the prediction
- sequences will be loaded from fasta (with one sequence)
"""
import json
from pathlib import Path
from collections import namedtuple
import numpy as np
import pandas as pd
import tensorflow as tf
from Bio import SeqIO
from tqdm import tqdm

from supervised_dna.saliency_maps import (
    get_saliencymap,
    get_kmer_importance,
)

from supervised_dna import ModelLoader
from supervised_dna.fcgr import FCGR
from supervised_dna.utils import (
    array2img,
    find_matches,
    fcgrpos2kmers,
    preprocess_seq
)

from parameters import PARAMETERS
KMER = PARAMETERS["KMER"]

# -1- Load model with best weights
loader = ModelLoader()
MODEL  = "resnet50_8mers"
CLADES = ['S','L','G','V','GR','GH','GV','GK']
WEIGHTS_PATH = "checkpoints/model-02-0.969.hdf5"
model  = loader(
            model_name=MODEL,
            n_outputs=len(CLADES),
            weights_path=WEIGHTS_PATH,
            ) # get compiled model from ./supervised_dna/models

# -2- Load sequences
# load list of npy for testing and then refer to the fasta file
with open("datasets.json","r") as fp: 
    datasets = json.load(fp)
list_test = datasets["test"]

def modify_path(path): 
    """Modify paths 'npy-8-mer/hCoV-19/{CLADE}/{seq_name}.npy' 
    -> 'data/hCoV-19/{CLADE}/{seq_name}.fasta' """
    path = path.replace("npy-8-mer","data")
    path = path.replace(".npy",".fasta")
    return path

list_test = [modify_path(path) for path in list_test]
available_data = [str(path) for path in Path("data").rglob("*.fasta")] #datasets["test"] # list of npy files 'npy-8-mer/hCoV-19/{CLADE}/{seq_name}.npy'
list_test = [path for path  in set(list_test).intersection(set(available_data))]

# Analysis
fcgr = FCGR(k=KMER)
pos2kmer = fcgrpos2kmers(k=KMER) # dict with position in FCGR to kmer
THRESHOLD_SALIENCYMAP = 0.1

# --- for one sample ---
# path_fasta = list_test[0]

def compute_analysis(path_fasta):
    fasta = next(SeqIO.parse(path_fasta, "fasta"))
    array_freq = fcgr(sequence=preprocess_seq(str(fasta.seq)))
    # preproceesing (divide by 10) and add channel axis
    input_model = np.expand_dims(array_freq/10. , axis=-1)
    input_model = np.expand_dims(input_model,axis=0)

    grad_eval = get_saliencymap(model, input_model)

    kmer_importance = get_kmer_importance(grad_eval, THRESHOLD_SALIENCYMAP, array_freq, pos2kmer)

    # obtain positions where each kmer match in the original sequence
    df = pd.DataFrame(kmer_importance)
    df["matches"] = df.apply(lambda row: find_matches(row["kmer"],str(fasta.seq)) if row["freq"]>0 else None,axis=1)

    # Save results: saliency_map
    path_grad_eval = str(path_fasta).replace("data","saliency-maps")
    path_grad_eval = path_grad_eval.replace(".fasta",".npy")
    Path(path_grad_eval).parent.mkdir(parents=True, exist_ok=True)
    np.save(path_grad_eval, grad_eval)

    # Save results: kmer importance
    path_kmer_importance = path_grad_eval = str(path_fasta).replace("data","kmer-importance")
    path_kmer_importance = path_grad_eval.replace(".fasta",".csv")
    Path(path_kmer_importance).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path_kmer_importance)
    # --- for one sample ----

for path_fasta in tqdm(list_test, desc="Computing Saliency Maps"):
    compute_analysis(path_fasta)