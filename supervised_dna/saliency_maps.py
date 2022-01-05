from collections import namedtuple
import numpy as np
import tensorflow as tf

def get_saliencymap(model, input_model): 
    """Compute gradients for each pixel w/r to the predicted class

    Args:
        model (tf.keras.Model): trained CNN model
        input_model (array or tensor): input for the model

    Returns:
        (array): gradients for each pixel normalized in [0,1]
    """    
    input_ = tf.Variable(input_model, dtype=float)

    with tf.GradientTape() as tape:
        pred = model(input_, training=False)
        class_idxs_sorted = np.argsort(pred.numpy().flatten())[::-1]
        loss = pred[0][class_idxs_sorted[0]]
        
    grads = tape.gradient(loss, input_)
    dgrad_abs = tf.math.abs(grads)
    dgrad_max_ = np.max(dgrad_abs, axis=3)[0] # i think this is not necessary for grayscale images

    ## normalize to range between 0 and 1
    arr_min, arr_max  = np.min(dgrad_max_), np.max(dgrad_max_)
    grad_eval = (dgrad_max_ - arr_min) / (arr_max - arr_min + 1e-18)

    return grad_eval

def get_kmer_importance(grad_eval, threshold, array_freq, pos2kmer):
    """find most relevant kmers for the prediction

    Args:
        grad_eval (array]): [description]
        threshold (float): between 0 and 1 to filter grad_eval
        array_freq (array): the FCGR matrix
        pos2kmer (dict): dictionary to map position in FCGR to the kmer

    Returns:
        list: information about kmer importance, sorted by the value of the gradient
    """    
    kmer_importance=namedtuple("kmer_importance", ["kmer","row","col","grad","freq"])

    rows, cols = np.where(grad_eval > threshold)
    list_saliency = []
    for row,col in zip(rows,cols):
        list_saliency.append(
            kmer_importance(pos2kmer.get((row,col)), 
            row,
            col, 
            grad_eval[row,col], 
            array_freq[row,col]
            )
        )

    # Sort values by importance (value of the gradient) 
    sorted(list_saliency, key=lambda x: x.grad,reverse=True)

    return list_saliency

