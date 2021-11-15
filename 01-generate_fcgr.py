from pathlib import Path
from parameters import PARAMETERS
from supervised_dna import (
    GenerateFCGR,
)

FOLDER_FASTA = Path(PARAMETERS["FOLDER_FASTA"]) 
LIST_FASTA   = list(FOLDER_FASTA.rglob("*fas"))
KMER = PARAMETERS["KMER"] 
BITS = PARAMETERS["BITS"]

#Instantiate class to generate FCGR
generate_fcgr = GenerateFCGR(
                destination_folder="img-{}-mer".format(KMER),
                kmer=KMER,
                bits=BITS,
                )

# Generate FCGR for a list of fasta files
generate_fcgr(list_fasta=LIST_FASTA,)