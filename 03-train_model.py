import json
from supervised_dna import (
    ModelLoader,
    DatasetLoader,    
)
from parameters import PARAMETERS, TRAIN
from supervised_dna.data_generator import DataGenerator
import tensorflow as tf


# General parameters
KMER = PARAMETERS["KMER"]

# Train parameters
BATCH_SIZE = TRAIN["BATCH_SIZE"]
EPOCHS     = TRAIN["EPOCHS"]

# -1- Model selection
loader = ModelLoader()
model  = loader("vgg16_{}mers".format(KMER)) # get compiled model from ./supervised_dna/models

# -2- Datasets
# load list of images for train and validation sets
with open("datasets.json","r") as f:
    datasets = json.load(f)
list_train = datasets["train"]
list_val   = datasets["val"][:64]

# prepare datasets to feed the model
config_generator = dict(
    order_output_model = ["1","2","3","4"],
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

# ds_loader = DatasetLoader(batch_size=BATCH_SIZE, 
#                             kmer=KMER, 
#                             order_output_model=["1","2","3","4"],
#                             shuffle=False,
#                             )
# ds_loader2 = DatasetLoader(batch_size=1, 
#                             kmer=KMER, 
#                             order_output_model=["1","2","3","4"],
#                             shuffle=False,
#                             )

# ds_train = ds_loader(list_img=list_train[:64])
# ds_val = ds_loader2(list_img=list_train[:64])

# ds = ds_loader(list_img=list_val) 
# image_count = len(ds)
# val_size = int(image_count * 0.2)
# ds_train = ds.skip(val_size)
# ds_val = ds.take(val_size)

# ds_val    = ds_loader(list_img=list_val[:20])
# ds_val2    = ds_loader2(list_img=list_val[:20])

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