from ultralytics import YOLO
from evaluation import constants
import onnxruntime as ort
import numpy as np

def   detection_gauge_face(img, model_path='yolov8n.pt', conf=0.25, optimized=False):
    '''
    uses yolo v8 to get bounding box of gauge face
    :param img: numpy image
    :param model_path: path to yolov8 detection model
    :return: highest confidence box for further processing and list of all boxes for visualization
    '''

    if not optimized:
        task = "detect"
        model = YOLO(model_path, task=task)  # load model
        # results = model(img)  # run inference, detects gauge face and needle
        results = model.predict(img, conf=conf)     # , imgsz=constants.ORIGINAL_IMG_SIZE)
        print(results[0])
    else:
        model = ort.InferenceSession(model_path, providers=ort.get_available_providers())
        input_name = model.get_inputs()[0].name
        print(input_name)
        img = img.astype(np.float32)
        img = img.transpose((2, 0, 1))
        img = np.expand_dims(img, 0)
        print(f"image type: {type(img)} {img.shape}")
        results = model.run(None, {input_name : img})
        print(results)

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

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

