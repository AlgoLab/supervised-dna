import json
import pandas as pd
from parameters import PARAMETERS
from supervised_dna import (
    ModelLoader,
    DataGenerator,
)

KMER = PARAMETERS["KMER"]
BATCH_SIZE = PARAMETERS["BATCH_SIZE"]
CLADES = PARAMETERS["CLADES"]
BITS = PARAMETERS["BITS"]

# -1- Load model
loader = ModelLoader()
model  = loader("cnn_{}mers".format(KMER), weights_path="checkpoint/cp.ckpt") # get compiled model from ./supervised_dna/models

# -2- Datasets
# load list of images for train and validation sets
with open("datasets.json","r") as f:
    datasets = json.load(f)
list_test = datasets["test"]

config_generator = dict(
    order_output_model = CLADES,
    batch_size = BATCH_SIZE,
    shuffle = False,
    kmer = KMER,
    bits = BITS,
)

ds_test = DataGenerator(
    list_test,
    **config_generator
) 

# Evaluate model and save metrics
result = model.evaluate(ds_test)
pd.DataFrame(
    dict(zip(model.metrics_names, result)), index=[0]) \
        .to_csv("metrics_test.csv")