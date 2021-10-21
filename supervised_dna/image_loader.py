from pathlib import Path
from typing import List
import tensorflow as tf
from .encoder_output import EncoderOutput

class ImageLoader:
    """Load Image"""
    def __init__(self, img_height, img_width):
        self.img_height = img_height
        self.img_width  = img_width

    def decode_img(self, img):
        "load image as raw data, then transform it to image"
        # convert the compressed string to a 2D uint8 tensor
        img = tf.io.decode_jpeg(img, channels=1)
        # resize the image to the desired size
        return tf.image.resize(img, [self.img_height, self.img_width])

    def load_img(self, file_path):
        img = tf.io.read_file(file_path)
        img = self.decode_img(img)
        return img

class InputOutputLoader(ImageLoader):

    def __init__(self, img_height, img_width, order_output_model: List[str]):
        super().__init__(img_height, img_width)
        self.order_output_model = order_output_model
        self.encoder_output = EncoderOutput(order_output_model)

    def __call__(self, file_path: Path,):
        "given an image path, return the input-output for the model"
        file_path = str(file_path)
        label = Path(file_path).parent.stem
        img = self.load_img(file_path)
        return img, self.encoder_output([label])
