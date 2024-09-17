import argparse
import os
import time
import sys
# import json
import subprocess
from pathlib import Path

import numpy as np
import torch
import onnx, onnxslim
import onnxruntime as ort
import tensorflow as tf
import onnx2tf
# from onnx_tf.backend import prepare

from model import load_model, N_HEATMAPS, INPUT_SIZE

BATCH_SIZE = 1
CHANNELS = 3


def main():
    args = read_args()

    model_path = args.model_path

    model = load_model(model_path)
    full_model_path = Path(str(model_path).replace(".pt", "_full.pt"))
    # torch.save(model, full_model_path)

    onnx_model_path = Path(str(model_path).replace(".pt", ".onnx"))
    print(f"onnx path: {onnx_model_path}")

    torch_input = torch.randn((BATCH_SIZE, CHANNELS, INPUT_SIZE[0], INPUT_SIZE[1]), dtype=torch.float32)
    # torch_output = model(torch_input)
    torch.onnx.export(model,
                      torch_input,
                      onnx_model_path,
                      opset_version=18,
                      input_names=['input'],
                      output_names=['output'])    
    print("Model saved to onnx")

    time.sleep(2)
    print("Checking onnx Model for errors")
    onnx_model = onnx.load(onnx_model_path)

    onnx_model = onnxslim.slim(onnx_model)
    onnx.checker.check_model(onnx_model)

    # ort_session = ort.InferenceSession(onnx_model)
    # outputs = ort_session.run(None, {'input': np.random.randn(BATCH_SIZE, CHANNELS, INPUT_SIZE[0], INPUT_SIZE[1]).astype(np.float32)})

    # print(f"output: {outputs}")
    print("Pytorch model successfully converted to onnx")

    time.sleep(2)
    print("Converting model to TensorFlow format")

    tf_model_path = Path(str(model_path).replace(".pt", "_saved_model"))
    print(f"TF model path: {tf_model_path}")

    verbosity = "debug"

    onnx2tf.convert(
        input_onnx_file_path=onnx_model_path,
        output_folder_path=tf_model_path,
        not_use_onnxsim=True,
        verbosity=verbosity,
        output_integer_quantized_tflite=True,
        quant_type="per-tensor",
        disable_group_convolution=True,
        enable_batchmatmul_unfold=True,
    )

    # Rename the dynamic range quant file to int8 file
    for file in tf_model_path.glob("*_dynamic_range_quant.tflite"):
        file.rename(file.with_name(file.stem.replace("_dynamic_range_quant", "_int8") + file.suffix))

    print("TF Lite Model Saved") 

    time.sleep(2)
    # print("Converting model to edgetpu format")

    tflite_model_path = Path(tf_model_path, Path(tf_model_path).stem.replace("_saved_model", "_full_integer_quant.tflite"))
    print(f"TF Lite Model Path: {tflite_model_path}")

    edgetpu_model_path = Path(str(tflite_model_path).replace(".tflite", "_edgetpu.tflite"))
    print(f"Edgetpu Model Path: {edgetpu_model_path}")


    cmd = f'edgetpu_compiler -s -d -k 10 -o "{Path(edgetpu_model_path).parent}" "{tflite_model_path}"'
    print(cmd)
    subprocess.run(cmd, shell=True)
    print("Edgetpu model saved")

    print("All Done")

def read_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',
                        type=str,
                        required=True,
                        help="path to pytorch model")

    return parser.parse_args()

if __name__ == '__main__':
    main()