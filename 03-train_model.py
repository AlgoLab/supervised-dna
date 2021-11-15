import json
from supervised_dna import (
    ModelLoader,
    DataGenerator,    
)
from parameters import PARAMETERS
import tensorflow as tf

# General parameters
KMER = PARAMETERS["KMER"]
CLADES = PARAMETERS["CLADES"]

# Train parameters
BATCH_SIZE = PARAMETERS["BATCH_SIZE"]
EPOCHS     = PARAMETERS["EPOCHS"]


# -1- Model selection
loader = ModelLoader()
model  = loader("cnn_{}mers".format(KMER)) # get compiled model from ./supervised_dna/models

# -2- Datasets
# load list of images for train and validation sets
with open("datasets.json","r") as f:
    datasets = json.load(f)
list_train = datasets["train"]
list_val   = datasets["val"]

# prepare datasets to feed the model
config_generator = dict(
    order_output_model = CLADES,
    shuffle = False,
)

# Instantiate DataGenerator for training set
ds_train = DataGenerator(
    list_val,
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