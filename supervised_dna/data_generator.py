"Load data to the model (Same as dataset.py but with keras)"

from typing import List, Union
from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image
from .encoder_output import EncoderOutput
from .image_loader import InputOutputLoader

 # Load min_array.jpg
BASE_DIR = Path(__file__).resolve().parent.parent
PATH_REF_ARRAY = BASE_DIR.joinpath("ref_array.jpg")
REF_ARRAY = np.asarray(Image.open(str(PATH_REF_ARRAY)))
REF_ARRAY = np.expand_dims(REF_ARRAY,axis=-1)

class DataGenerator(tf.keras.utils.Sequence):
    """Data Generator  for keras from a list of paths to files""" 

    def __init__(self, 
                list_paths: List[Union[str, Path]], 
                order_output_model: List[str],
                batch_size: int = 8,
                shuffle: bool = True,
                kmer: int = 8,
                bits: int = 8,
                
                ):
        self.list_paths = list_paths
        self.order_output_model = order_output_model   
        self.batch_size = batch_size 
        self.shuffle = shuffle
        self.kmer = kmer
        self.bits = bits
        self.input_output_loader = InputOutputLoader(2**kmer, 2**kmer, order_output_model)
        
        # initialize first batch
        self.on_epoch_end()

    def on_epoch_end(self,):
        """Updates indexes after each epoch (starting for the epoch '0')"""
        self.indexes = np.arange(len(self.list_paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes) # shuffle indexes in place

    def __len__(self):
        # Must be implemented
        """Denotes the number of batches per epoch"""
        delta = 1 if len(self.list_paths) % self.batch_size else 0 
        return len(self.list_paths) // self.batch_size + delta

    def __getitem__(self, index):
        # Must be implemented
        """To feed the model with data in training
        It generates one batch of data
        """
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of paths to ecg
        list_paths_temp = [self.list_paths[k] for k in indexes]

        # Generate data
        X, y = self.input_output_generation(list_paths_temp)
        return X, y
    
    def input_output_generation(self, list_path_temp: List[str]): 
        """Generates and augment data containing batch_size samples
        Args:
            list_path_temp (List[str]): sublist of list_path
        Returns:
            X : numpy.array
            y : numpy.array hot-encoding
        """ 
        X_batch = []
        y_batch = []
        for path in list_path_temp: 
            img, label = self.input_output_loader(path)
            img = self.preprocessing(img)
            X_batch.append(np.expand_dims(img,axis=0)) # add to list with batch dims
            y_batch.append(label)

        return np.concatenate(X_batch, axis=0), np.array(y_batch)

    def preprocessing(self, img):
        "The inpt img is loaded as a (2**K,2**K,1) dimensional array"
        # Substract min_array
        img = img - REF_ARRAY
        # Scale to [0,1]
        img /= float(2**self.bits-1)
        return img