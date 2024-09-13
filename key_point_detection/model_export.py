import argparse
import os
import time
import sys
import json

import numpy as np
import torch
import onnx, onnxslim

from model import load_model, N_HEATMAPS, INPUT_SIZE


def main():
    args = read_args()

    model_path = args.model_path

    model = load_model(model_path)
    full_model_path = str(model_path).replace(".pt", "_full.pt")
    torch.save(model, full_model_path)

    onnx_model_path = str(model_path).replace(".pt", ".onnx")
    print(f"onnx path: {onnx_model_path}")

    torch_input = torch.randn((1, 3, INPUT_SIZE[0], INPUT_SIZE[1]), dtype=torch.float32)
    # torch_output = model(torch_input)
    torch.onnx.export(model,
                      (torch_input),
                      onnx_model_path)    
    print("Model saved to onnx")

    time.sleep(2)
    print("Checking onnx Model for errors")
    onnx_model = onnx.load(onnx_model_path)

    onnx_model = onnxslim.slim(onnx_model)
    onnx.checker.check_model(onnx_model)
    print(f"All done")

def read_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',
                        type=str,
                        required=True,
                        help="path to pytorch model")

    return parser.parse_args()

if __name__ == '__main__':
    main()