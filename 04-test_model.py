import json
import pandas as pd
from parameters import PARAMETERS
from supervised_dna import (
    ModelLoader,
    DatasetLoader,
)

KMER = PARAMETERS["KMER"]
BATCH_SIZE = 8

# -1- Load model
loader = ModelLoader()
model  = loader("cnn_{}mers".format(KMER), weights_path="checkpoint/cp.ckpt") # get compiled model from ./supervised_dna/models

# -2- Datasets
# load list of images for train and validation sets
with open("datasets.json","r") as f:
    datasets = json.load(f)
list_test = datasets["val"][:64]


ds_loader = DatasetLoader(batch_size=BATCH_SIZE, 
                            kmer=KMER, 
                            order_output_model=["1","2","3","4"],
                            shuffle=False,
                            )

ds_test = ds_loader(list_img = list_test)

# Evaluate model and save metrics
result = model.evaluate(ds_test)
pd.DataFrame(
    dict(zip(model.metrics_names, result)), index=[0]) \
        .to_csv("metrics_test.csv")