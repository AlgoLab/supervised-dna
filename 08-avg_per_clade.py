"Compute average kmer-frequency/ saliency-map/ kmer-importance per clade"
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm 

import matplotlib.pyplot  as plt
from supervised_dna.utils import fcgrpos2kmers, array2img
from supervised_dna.saliency_maps import get_kmer_importance

from parameters import PARAMETERS

k = PARAMETERS["KMER"]
CLADES = PARAMETERS["CLADES"]

# results on prediction
path_model_preds = "results_nextclade_comparison.csv"
model_preds = pd.read_csv(path_model_preds)

# get ground truth for the sequence
model_preds["ground_truth"] = model_preds["path_fasta"].apply(lambda path: path.split("/")[2])
model_preds["TP"] = model_preds.apply(lambda row: row["ground_truth"]==row["pred_class"], axis=1)

for clade in tqdm(CLADES):
    path_save = Path(f"avg-results-nextclade-comparison/{clade}")
    path_save.mkdir(parents=True, exist_ok=True)

    # filter only True Positive samples 
    paths_TP_clade = model_preds.query(f"`ground_truth` == '{clade}' and `TP`==True")["path_fasta"].tolist()

    # saliency maps
    list_saliencymaps = [
        path.replace("data","saliency-maps").replace(".fasta",".npy")
                                                    for path in paths_TP_clade
                                                    ]

    # kmer importance
    list_freqkmer = [
        path.replace("data","freq-kmer").replace(".fasta",".npy")
                                                    for path in paths_TP_clade
                                                    ]
    # list_saliencymaps = list(Path(f"saliency-maps-nextclade-comparison/hCoV-19/{clade}").rglob("*.npy"))
    # list_freqkmer = list(Path(f"freq-kmer-nextclade-comparison/hCoV-19/{clade}").rglob("*.npy"))

    # average saliency maps
    sm_clade = np.zeros((2**k,2**k)) # saliency map
    for path_saliencymap in list_saliencymaps: 
        sm = np.load(path_saliencymap)
        sm_clade = np.add(sm_clade, sm)
    avg_sm = sm_clade / len(list_saliencymaps)
    np.save(path_save.joinpath("saliency_map.npy"), avg_sm)

    # average kmer frequency
    freq_clade = np.zeros((2**k,2**k)) # freq kmers
    for path_freqkmer in list_freqkmer: 
        freq = np.load(path_freqkmer)
        freq_clade = np.add(freq_clade, freq)
    avg_freq = freq_clade / len(list_freqkmer)
    np.save(path_save.joinpath("kmer_frequency.npy"), avg_freq)

    # k-mer importance  
    pos2kmer = fcgrpos2kmers(k=k)
    kmer_importance = get_kmer_importance(avg_sm, 
                        threshold=0.2, 
                        array_freq=avg_freq, 
                        pos2kmer=pos2kmer
                        )

    avg_kmer_importance = pd.DataFrame(kmer_importance)
    avg_kmer_importance.to_csv(path_save.joinpath("kmer_importance.csv"))
