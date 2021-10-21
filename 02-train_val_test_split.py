import json
import pandas as pd
from parameters import PARAMETERS
from pathlib import Path
from supervised_dna import DataSelector

# Select all data
FOLDER_IMG = Path(PARAMETERS["FOLDER_IMG"]) 
LIST_FASTA   = list(FOLDER_IMG.rglob("*jpg"))
TRAIN_SIZE   = float(PARAMETERS["TRAIN_SIZE"]) 

# Input for DataSelector
id_labels = [str(path) for path in LIST_FASTA]
labels    = [path.parent.stem for path in LIST_FASTA]

# Instantiate DataSelector
ds = DataSelector(id_labels, labels)

# Get train, test and val sets
ds(train_size=TRAIN_SIZE, balanced_on=labels)


with open("datasets.json", "w", encoding="utf-8") as f: 
    json.dump(ds.datasets["id_labels"], f, ensure_ascii=False, indent=4)

# Summary of data selected 
summary_labels =  pd.DataFrame(ds.get_summary_labels())
summary_labels["Total"] = summary_labels.sum(axis=1)
summary_labels.to_csv("summary_labels.csv")