import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class LayerApplyMaskTensor(tf.keras.layers.Layer):

    def __init__(self, mask_tensor, name='mask'):
        super(LayerApplyMaskTensor, self).__init__(name=name)
        self.mask_tensor = mask_tensor

    def call(self, input):
        output = tf.math.multiply(input, self.mask_tensor)
        return output


