import tensorflow as tf
import numpy as np
import scipy.io
import os
import sys
import argparse
sys.path.append('../')
from model import Attention_mask, MTTS_CAN
import h5py
import matplotlib.pyplot as plt
from scipy.signal import butter
from inference_preprocess import preprocess_raw_video, detrend

def predict_vitals(args):
    img_rows = 36
    img_cols = 36
    frame_depth = 10
    model_checkpoint = './mtts_can.hdf5'
    batch_size = args.batch_size
    fs = args.sampling_rate
    sample_data_path = args.video_path

    dXsub = preprocess_raw_video(sample_data_path, dim=36)
    print('dXsub shape', dXsub.shape)

    dXsub_len = (dXsub.shape[0] // frame_depth) * frame_depth
    dXsub = dXsub[:dXsub_len, :, :, :]

    #model = MTTS_CAN(frame_depth, 32, 64, (img_rows, img_cols, 3))
    #model.load_weights(model_checkpoint)

    interpreter = tf.lite.Interpreter(model_path="C:/Users/anand/Documents/Current/Pulse/MTTS-CAN/code/model.tflite")
    input_details = interpreter.get_input_details()
    print("Before: " + str(input_details))
    interpreter.resize_tensor_input(input_details[0]['index'], [dXsub_len, img_rows, img_cols, 3])
    interpreter.resize_tensor_input(input_details[1]['index'], [dXsub_len, img_rows, img_cols, 3])
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    print("After: " + str(input_details))
    output_details = interpreter.get_output_details()
    print("Output details: " + str(output_details))

    interpreter.set_tensor(input_details[0]['index'], dXsub[:, :, :, :3])
    interpreter.set_tensor(input_details[1]['index'], dXsub[:, :, :, -3:])
    interpreter.invoke()
    pulse_pred = interpreter.get_tensor(output_details[0]['index'])
    resp_pred = interpreter.get_tensor(output_details[1]['index'])

    pulse_pred = detrend(np.cumsum(pulse_pred), 100)
    [b_pulse, a_pulse] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')
    pulse_pred = scipy.signal.filtfilt(b_pulse, a_pulse, np.double(pulse_pred))

    resp_pred = detrend(np.cumsum(resp_pred), 100)
    [b_resp, a_resp] = butter(1, [0.08 / fs * 2, 0.5 / fs * 2], btype='bandpass')
    resp_pred = scipy.signal.filtfilt(b_resp, a_resp, np.double(resp_pred))

    ########## Plot ##################
    plt.subplot(211)
    plt.plot(pulse_pred)
    plt.title('Pulse Prediction')
    plt.subplot(212)
    plt.plot(resp_pred)
    plt.title('Resp Prediction')
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, help='processed video path')
    parser.add_argument('--sampling_rate', type=int, default = 30, help='sampling rate of your video')
    parser.add_argument('--batch_size', type=int, default = 100, help='batch size (multiplier of 10)')
    args = parser.parse_args()

    predict_vitals(args)

