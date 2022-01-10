import streamlit as st
import time

from pathlib import Path
from urllib.error import URLError
from collections import namedtuple
import matplotlib.pyplot as plt

from Bio import SeqIO
import numpy as np
import pandas as pd
import tensorflow as tf

from supervised_dna import ModelLoader
from supervised_dna.fcgr import FCGR
from supervised_dna.saliency_maps import (
    get_saliencymap,
    get_kmer_importance,
)
from supervised_dna.utils import (
    array2img,
    find_matches,
    fcgrpos2kmers,
    preprocess_seq
)

def plot(array_freq, grad_eval):
    fig, axes = plt.subplots(1,2,figsize=(14,5))
    axes[0].imshow(array2img(array_freq), cmap="gray")
    i = axes[1].imshow(grad_eval,cmap="jet",alpha=0.8)
    fig.colorbar(i)
    return fig

def plot_kmer(df):
    ax = df.plot(x="kmer",y="grad", kind="bar")
    return ax.get_figure()
    

# --  Parameters experiment --
from parameters import PARAMETERS
KMER = PARAMETERS["KMER"]
MODEL  = "resnet50_8mers"
CLADES = ['S','L','G','V','GR','GH','GV','GK']
WEIGHTS_PATH = "checkpoints/model-02-0.969.hdf5"
THRESHOLD_SALIENCYMAP = 0.1

# Load Model
@st.cache(allow_output_mutation=True)
def load_model(): 
    loader=ModelLoader()
    model  = loader(
            model_name=MODEL,
            n_outputs=len(CLADES),
            weights_path=WEIGHTS_PATH,
            ) # get compiled model from ./supervised_dna/models
    return model    

# FCGR(position) to kmer
pos2kmer = fcgrpos2kmers(k=KMER)

CLADE ="S"

# Load data available
@st.cache(persist=True)
def read_path_fasta():
    return list(Path("data/hCoV-19/{}".format(CLADE)).rglob("*.fasta"))

# Load FCGR
fcgr = FCGR(k=KMER)

try:    
    # Inputs: 
    # - threshold for saliency maps
    threshold = st.slider(label="THRESHOLD SALIENCY MAP",
                    min_value=0.,
                    max_value=1.,
                    value=0.1,
                    step=0.1) 
    # sequence (fasta) to analyze
    filename  = st.selectbox(
        "Choose fasta", read_path_fasta(),
    )

    # Load model
    model = load_model()


    # Model Evaluation
    # Load and prepare input for the model   
    fasta = next(SeqIO.parse(filename, "fasta"))
    array_freq = fcgr(sequence=preprocess_seq(str(fasta.seq)))
    # preproceesing (divide by 10) and add channel axis
    input_model = np.expand_dims(array_freq/10. , axis=-1)
    input_model = np.expand_dims(input_model,axis=0)

    # Saliency Map
    grad_eval = get_saliencymap(model, input_model)
    
    # k-mer importance  
    kmer_importance = get_kmer_importance(grad_eval, 
                        threshold, 
                        array_freq, 
                        pos2kmer
                        )

    # obtain positions where each kmer match in the original sequence
    df = pd.DataFrame(kmer_importance)
    get_matches = lambda row, fasta: find_matches(row["kmer"],str(fasta.seq),return_str=True)
    df["matches"] = df.apply(lambda row: get_matches(row,fasta) if row["freq"]>0 else None,axis=1)

    
    fig = plot(array_freq,grad_eval)
    st.pyplot(fig)

    st.write("kmer importance", df)

    # 
    ByMatch = namedtuple("ByMatch",["match","grad","kmer"])
    list_by_match=[]
    for row in df.itertuples():
        if row.matches:
            matches = row.matches.split(",")
            kmer = row.kmer
            grad = row.grad
            for match in matches: 
                list_by_match.append(ByMatch(match,grad,kmer))            
    df_by_match = pd.DataFrame(list_by_match)
    df_by_match["match"] = df_by_match["match"].astype(int)
    df_by_match.sort_values(by="match",inplace=True,ignore_index=True)
    st.write("by match", df_by_match)
    bar_kmers = plot_kmer(df_by_match)
    st.pyplot(fig = bar_kmers)
    
except URLError as e:
    st.error(
        """
        **This demo requires internet access.**

        Connection error: %s
    """
        % e.reason
    )
# Streamlit widgets automatically run the script from top to bottom. Since
# this button is not connected to any other logic, it just causes a plain
# rerun.


st.button("Re-run")