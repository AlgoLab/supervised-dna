"""
SET PARAMETERS FOR ALL STEPS
"""
# -- Define parameters
# General
KMER = 8
SPECIE = "Bacteria"
# For training
TRAIN_SIZE = 0.8 # size for val and test sets = (1-TRAIN_SIZE)/2
BATCH_SIZE = 8
EPOCHS = 20

# ---------------------
# Load to a Dictionary
PARAMETERS = dict(
    KMER = KMER,
    FOLDER_FASTA = f"data/{SPECIE}",
    FOLDER_IMG = f"img-{KMER}-mer/{SPECIE}",
    TRAIN_SIZE = 0.8,
)

TRAIN = dict(
    BATCH_SIZE = BATCH_SIZE,
    EPOCHS = EPOCHS,       
)