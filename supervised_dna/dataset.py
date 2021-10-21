"Load data to the model (Same as data_generator.py but with tf.data.Dataset)"
from pathlib import Path
from typing import List
import tensorflow as tf
from .encoder_output import (
    EncoderOutput
)

AUTOTUNE = tf.data.AUTOTUNE

class DatasetLoader:
    """Load dataset from a list of paths
    Labels are inferred from the path
    """  
    def __init__(self, batch_size: int, 
                        kmer: int,
                        order_output_model: List[str], 
                        shuffle: bool = True):
        self.batch_size = batch_size
        self.kmer = kmer
        self.encoder_output = EncoderOutput(order_output_model)
        self.shuffle = shuffle

    def load_img_label(self, file_path, label):
        "return image loaded and label"
        img = tf.io.read_file(file_path)
        img = tf.io.decode_jpeg(img, channels=1,)
        return img, label

    def charge_dataset(self, list_img):
        "load path to images and extract labels (hot-encode) from path"
        labels = [self.encoder_output(Path(_).parent.stem) for _ in list_img]
        ds = tf.data.Dataset.from_tensor_slices((list_img,labels))
        n_files = len(ds)
        print("Total files: {}".format(n_files))
        return ds
    
    def configure_for_performance(self, ds): 
        # Performance of datasets
        ds = ds.cache()
        if self.shuffle is True: 
            ds = ds.shuffle(buffer_size=len(ds))
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    def preprocessing(self, ds):
        "Preprocess tf dataset"
        normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
        ds = ds.map(lambda img, label: (normalization_layer(img), label))
        return ds

    def __call__(self, list_img):
        
        # load list of files 
        ds = self.charge_dataset(list_img)

        # get img and labels from path
        ds = ds.map(self.load_img_label)
        
        # configure performance
        ds = self.configure_for_performance(ds)

        # preprocessing: normalize [0,1]
        ds = self.preprocessing(ds)

        return ds
