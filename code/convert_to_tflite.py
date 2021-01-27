from model import MTTS_CAN, TSM, TSM_Cov2D, Attention_mask
import tensorflow as tf
from tensorflow.python.keras import backend as K
import os
import sys

def convert_to_tflite():
    keras_model = tf.keras.models.load_model('C:/Users/anand/Documents/Current/Pulse/MTTS-CAN/mtts_can.hdf5', custom_objects={'MTTS_CAN': MTTS_CAN, 'TSM_Cov2D': TSM_Cov2D, 'TSM': TSM, 'Attention_mask': Attention_mask})
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)

    tflite_model = converter.convert()

    # save the model
    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)
    return

if __name__ == "__main__":
    convert_to_tflite()

