import cv2
import numpy as np
import os
import subprocess
import time
import argparse
import sys

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gauge_detection.detection_inference_onnx import detection_gauge_face, find_center_bbox
from metadata import write_metadata

CAMERA_NAME = "HD USB Camera"
# CAMERA_NAME = "HD Pro Webcam"

IMAGE_SIZE = (
    640, 640
)

MODEL_PATH = "models/gauge_detection_model_custom_trained.onnx"

def camera_adjust(camera_id, detection_model_path, metadata_path, debug=False):
    # Open the camera (0 is typically the default camera)
    # if camera_id == -1:
    #     command = f"v4l2-ctl --list-devices | grep '{CAMERA_NAME}' -A4 | sed -n '2p' | xargs"
    #     camera_id = subprocess.getstatusoutput(command)[1]
    #     if debug:
    #         print(f"index is {camera_id}")

    cap = cv2.VideoCapture(camera_id)
    # Set Camera properties
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 100)

    if cap.isOpened():
        # Start Timer for inference
        start_time = time.time()
        # Read frame
        success, frame = cap.read()

        if not success:
            return None

        # Resize the frame to 640 x 640
        frame = cv2.resize(frame, dsize=IMAGE_SIZE, interpolation=cv2.INTER_CUBIC)
        if debug:
            print(f"original image size: {frame.shape}")
        
        all_boxes = []

        # Formatting settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        box_color = (0, 255, 0)
        text_color = (0, 0, 255)
        box_thickness = 2
        text_thickness = 2

        try:
            # Convert the image to np array
            image = np.asarray(frame)
            all_boxes = detection_gauge_face(image, model_path=detection_model_path, conf=0.25)
        except Exception as err:
                err_msg = f"Unexpected {err=}, {type(err)=}"
                print(err_msg)
                pass
        finally:
            for index, box in enumerate(all_boxes):
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), box_color, box_thickness)
                cv2.putText(frame, f"#{index}", find_center_bbox(box), font, font_scale, text_color, text_thickness)

        end_time = time.time()
        inference_time = np.round(end_time - start_time, 2)
        if debug:
            print(f"Inference time: {inference_time} seconds")
        
        write_metadata(camera_id=camera_id, meter_configuration=[], metadata_file_path=metadata_path)
        return frame, inference_time
    cap.release()
    # cv2.destroyAllWindows()    
    print("Error in Camera ID; Please check the ID")
    return None

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--detection_model',
                        type=str,
                        required=False,
                        default=MODEL_PATH,
                        help="Path to detection model")
    parser.add_argument('--camera_id',
                        type=str,
                        required=False,
                        default=-1,
                        help="Camera ID")
    parser.add_argument('--metadata',
                        type=str,
                        required=False,
                        default="metadata.json",
                        help="Path to metadata file")
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()

if __name__ == "__main__":
    args = read_args()
    print(f"Camera ID: {args.camera_id}")
    while True:
        frame, inference_time = camera_adjust(int(args.camera_id), args.detection_model, args.metadata, args.debug)
        if frame is not None:
            cv2.imshow("feed", frame)
            print(f"Inference Time: {inference_time} s")
            key = cv2.waitKey(0)  & 0xFF # Wait for a key press to close the window        
            if key == ord('q'):
                cv2.destroyAllWindows()
                break
