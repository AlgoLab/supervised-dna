"""
SET PARAMETERS FOR ALL STEPS
"""
# -- Define parameters
# General
KMER = 8
SPECIE = "hCoV-19"
CLADES = ["GK","GV"]#['S','L','G','V','GR','GH','GV','GK']

# For training
TRAIN_SIZE = 0.8 # size for val and test sets = (1-TRAIN_SIZE)/2
BATCH_SIZE = 8
EPOCHS = 20
MODEL = "vgg16_8mers"

# ---------------------
# Load to a Dictionary
PARAMETERS = dict(
    KMER = KMER,
    CLADES = CLADES,
    FOLDER_FASTA = f"data/{SPECIE}",
    FOLDER_NPY = f"npy-{KMER}-mer/{SPECIE}",
    TRAIN_SIZE = 0.8,
    BATCH_SIZE = BATCH_SIZE,
    EPOCHS = EPOCHS,   
    MODEL = MODEL    
)