import random
from collections import namedtuple
from pathlib import Path
from tqdm import tqdm
import pandas as pd

from parameters import PARAMETERS

tqdm.pandas()
random.seed(42)

print(">> Undersample sequences <<")

PATH_METADATA = PARAMETERS["PATH_METADATA"]
CLADES = PARAMETERS["CLADES"]
SAMPLES_PER_CLADE = PARAMETERS["SAMPLES_PER_CLADE"]

# Load metadata
COLS = ["Virus name", "Accession ID", "Collection date", "Submission date","Clade", "Host", "Is complete?"]
data = pd.read_csv(PATH_METADATA,sep="\t")

# Remove NaN in Clades and not-complete sequences
data.dropna(axis="rows",
            how="any",
            subset=["Is complete?", "Clade"], 
            inplace=True,
            )

# Filter by Clades and Host
CLADES = tuple(clade for clade in CLADES)
data.query(f"`Clade` in {CLADES} and `Host`=='Human'", inplace=True)

## Randomly select a subset of sequences
# Generate id of sequences in fasta file: "Virus name|Accession ID|Collection date"
data["fasta_id"] = data.progress_apply(lambda row: "|".join([row["Virus name"],row["Collection date"],row["Submission date"]]), axis=1)

# subsample 
SampleClade = namedtuple("SampleClade", ["fasta_id","clade"])
list_fasta_selected = []
for clade in tqdm(CLADES):
    samples_clade = data.query(f"`Clade` == '{clade}'")["fasta_id"].tolist()
    random.shuffle(samples_clade)
    # select 'SAMPLES_PER_CLADE' samples for each clade, or all of them if available samples are less than required
    list_fasta_selected.extend([SampleClade(fasta_id, clade) for fasta_id in samples_clade[:SAMPLES_PER_CLADE]])

Path("data/train").mkdir(exists_ok=True, parents=True)
pd.DataFrame(list_fasta_selected).to_csv("data/train/undersample_by_clade.csv")