"""
Compute a reference image from the training set
"""
import json 
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from parameters import PARAMETERS

# -1- Load training images
with open("datasets.json","r") as f:
    datasets = json.load(f)
list_train = datasets["train"]
KMER = PARAMETERS["KMER"]

print("All images: ", len(list_train))

# -2- Load each image at a time and save the minimum value of pixels
path_img = list_train[0]

# Start with one array
npy = np.load(path_img)
# average image
ref_array = np.asarray(npy)

# minimum image
min_array = np.asarray(npy)

# load other images and sum values
for path_npy in tqdm(list_train[1:], desc="Computing avg image"):#, total=len(list_train)): 
    npy = np.load(path_npy)
    array = np.asarray(npy)
    # compute element-wise minimum between the new array and the current min_array
    ref_array = np.add(array, ref_array)

    min_array = np.minimum(array, min_array)

# compute avereage of arrays
ref_array = ref_array/len(list_train)

# save min_array
np.save("ref_array", ref_array)
np.save("min_array", min_array)