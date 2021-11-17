"""
Compute a reference image from the training set
"""
import json 
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from parameters import PARAMETERS
BITS = PARAMETERS["BITS"]
# -1- Load training images
with open("datasets.json","r") as f:
    datasets = json.load(f)
list_train = datasets["train"]
KMER = PARAMETERS["KMER"]

print("All images: ", len(list_train))

# -2- Load each image at a time and save the minimum value of pixels
path_img = list_train[0]

# Start with one image
img = Image.open(path_img)
ref_array = np.asarray(img)

# load other images and sum values
for path_img in tqdm(list_train[1:], desc="Computing avg image"):#, total=len(list_train)): 
    img = Image.open(path_img)
    array = np.asarray(img)
    # compute element-wise minimum between the new array and the current min_array
    ref_array = np.add(array, ref_array)

ref_array = ref_array/float(2*BITS)
# save min_array
Image.fromarray(ref_array,"L").save("ref_array.jpg")