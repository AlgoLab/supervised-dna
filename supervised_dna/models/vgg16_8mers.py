from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D, 
    Input, 
    Flatten, 
    Dense, 
    Dropout,
    concatenate
    )
# Reference name of model
MODEL_NAME = str(Path(__file__).resolve().stem)

# config
k = 8 # (256x256)

def get_model():
    input_model = (256,256,1)
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_model))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(4, activation='softmax', dtype = tf.float32))
    # Finally, we add a classification layer.
    optimizer = "Adam"
    model.compile(optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    return model


# def set_model_weights(model: models.Model, weight_list):
#     for i, symbolic_weights in enumerate(model.weights):
#         weight_values = weight_list[i]
#         K.set_value(symbolic_weights, weight_values)
