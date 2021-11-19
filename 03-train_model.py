import json
from pathlib import Path
import numpy as np
import tensorflow as tf

from supervised_dna import (
    ModelLoader,
    DataGenerator,    
)
from parameters import PARAMETERS

# General parameters
KMER = PARAMETERS["KMER"]
CLADES = PARAMETERS["CLADES"]

# Train parameters
BATCH_SIZE = PARAMETERS["BATCH_SIZE"]
EPOCHS     = PARAMETERS["EPOCHS"]
MODEL      = PARAMETERS["MODEL"]
# -1- Model selection
loader = ModelLoader()
model  = loader(
            model_name=MODEL,
            n_outputs=len(CLADES)
            ) # get compiled model from ./supervised_dna/models

# -2- Datasets
# load list of images for train and validation sets
with open("datasets.json","r") as f:
    datasets = json.load(f)
list_train = datasets["train"]
list_val   = datasets["val"]

# - Preprocessing -
#  # Load min_array.jpg
# BASE_DIR = Path(__file__).resolve().parent.parent
# PATH_REF_ARRAY = BASE_DIR.joinpath("ref_array.npy")
# REF_ARRAY = np.load(str(PATH_REF_ARRAY))
# REF_ARRAY = np.expand_dims(REF_ARRAY,axis=-1)
# MAX_VALUE = REF_ARRAY.max()

def preprocessing(self, npy, ref_array, max_value):
    "The input npy is loaded as a (2**K,2**K,1) dimensional array"
    # Substract min_array
    npy = np.subtract(npy,ref_array)

    # Scale around/approx [0,1]
    npy /= max_value
    return npy

# prepare datasets to feed the model
config_generator = dict(
    order_output_model = CLADES,
    batch_size = BATCH_SIZE,
    shuffle = False,
    kmer = KMER,
    preprocessing = preprocessing,
)

# Instantiate DataGenerator for training set
ds_train = DataGenerator(
    list_train,
    **config_generator
)

ds_val = DataGenerator(
    list_val,
    **config_generator
) 

# -3- Training
# Callbacks
checkpoint_path = "checkpoint/cp.ckpt"

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    save_weights_only=True,
    monitor='loss',
    mode='min',
    save_best_only=True)

model.fit(
    ds_train,
    validation_data=ds_val,
    epochs=EPOCHS,
    callbacks=[
        model_checkpoint_callback,
        tf.keras.callbacks.EarlyStopping(patience=5),
        ]
)