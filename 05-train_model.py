import os
import json
from pathlib import Path
import numpy as np
import tensorflow as tf

# This works for limit the number of threads
# https://github.com/tensorflow/tensorflow/issues/29968
num_threads = 1
# Maximum number of threads to use for OpenMP parallel regions.
os.environ["OMP_NUM_THREADS"] = "1"
# Without setting below 2 environment variables, it didn't work for me. Thanks to @cjw85 
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

tf.config.threading.set_inter_op_parallelism_threads(
    num_threads
)
tf.config.threading.set_intra_op_parallelism_threads(
    num_threads
)
tf.config.set_soft_device_placement(True)


from supervised_dna import (
    ModelLoader,
    DataGenerator,    
)
from parameters import PARAMETERS

SEED = PARAMETERS["SEED"]
tf.random.set_seed(SEED)

# General parameters
KMER = PARAMETERS["KMER"]
CLADES = PARAMETERS["CLADES"]

# Train parameters
BATCH_SIZE = PARAMETERS["BATCH_SIZE"]
EPOCHS     = PARAMETERS["EPOCHS"]
MODEL      = PARAMETERS["MODEL"]

#WEIGHTS_PATH = "checkpoints/model-02-0.969.hdf5"

with tf.device('/CPU:0'):
    # -1- Model selection
    loader = ModelLoader()
    model  = loader(
                model_name=MODEL,
                n_outputs=len(CLADES),
                #weights_path=WEIGHTS_PATH
                ) # get compiled model from ./supervised_dna/models

    # -2- Datasets
    # load list of images for train and validation sets
    with open("datasets.json","r") as f:
        datasets = json.load(f)
    list_train = datasets["train"]
    list_val   = datasets["val"]

    def preprocessing(npy):
        "The input npy is loaded as a (2**K,2**K,1) dimensional array"
        # Scale around/approx [0,1]
        npy /= 10.
        return npy

    ## prepare datasets to feed the model
    # Instantiate DataGenerator for training set
    ds_train = DataGenerator(
        list_train,
        order_output_model = CLADES,
        batch_size = BATCH_SIZE,
        shuffle = True,
        kmer = KMER,
        preprocessing = preprocessing,
    )

    # Instantiate DataGenerator for validation set
    ds_val = DataGenerator(
        list_val,
        order_output_model = CLADES,
        batch_size = BATCH_SIZE,
        shuffle = False,
        kmer = KMER,
        preprocessing = preprocessing,
    ) 

    # -3- Training
    # Callbacks

    # checkpoint: save best weights
    Path("data/train/checkpoints").mkdir(exist_ok=True, parents=True)
    cb_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath='data/train/checkpoints/model-{epoch:02d}-{val_accuracy:.3f}.hdf5',
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=1
    )

    # reduce learning rate
    cb_reducelr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        mode='min',
        factor=0.1,
        patience=8,
        verbose=1,
        min_lr=0.00001
    )

    # stop training if
    cb_earlystop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        mode='min',
        min_delta=0.001,
        patience=10,
        verbose=1
    )

    # save history of training
    Path("data/train").mkdir(exist_ok=True, parents=True)
    cb_csvlogger = tf.keras.callbacks.CSVLogger(
        filename='data/train/training_log.csv',
        separator=',',
        append=False
    )

    model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=EPOCHS,
        callbacks=[
            cb_checkpoint,
            cb_reducelr,
            cb_earlystop,
            cb_csvlogger,
            ]
    )