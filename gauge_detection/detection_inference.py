from ultralytics import YOLO
from evaluation import constants
import numpy as np

def   detection_gauge_face(img, model_path='yolov8n.pt', conf=0.25):
    '''
    uses yolo v8 to get bounding box of gauge face
    :param img: numpy image
    :param model_path: path to yolov8 detection model
    :return: highest confidence box for further processing and list of all boxes for visualization
    '''
    task = "detect"
    model = YOLO(model_path, task=task)  # load model
    results = model.predict(img, conf=conf)     # , imgsz=constants.ORIGINAL_IMG_SIZE)

    # get list of detected boxes, already sorted by confidence
    boxes = results[0].boxes

    if len(boxes) == 0:
        raise Exception("No gauge detected in image")

    # get highest confidence box which is of a gauge face
    # gauge_face_box = boxes[0]

    box_list = []
    for box in boxes:
        box_list.append(box.xyxy[0].int())

    return box_list

def find_center_bbox(box):
    center = (int(box[0]) + (int(box[2] - box[0]) // 2), int(box[1]) + (int(box[3] - box[1]) // 2))
    return center
