from ultralytics import YOLO
from evaluation import constants
# import onnxruntime as ort
import numpy as np

def detection_gauge_face(img, model_path='yolov8n.pt', conf=0.25, optimized=False):
    '''
    uses yolo v8 to get bounding box of gauge face
    :param img: numpy image
    :param model_path: path to yolov8 detection model
    :return: highest confidence box for further processing and list of all boxes for visualization
    '''

    if not optimized:
        task = "detect"
        model = YOLO(model_path, task=task)  # load model
        results = model.predict(img, conf=conf)     # , imgsz=constants.ORIGINAL_IMG_SIZE)
        # print(results[0])
    # else:
        # Add mechanism to run ort without YOLO
        # model = ort.InferenceSession(model_path, providers=ort.get_available_providers())
        # # input_name = model.get_inputs()[0].name
        # # print(input_name)
        # img = preprocess(img)
        # img_size = np.array([640, 640], dtype=np.float32).reshape(1, 2)
        # results = model.run(None, {"images": img})[0]
    
        # print(f"results: {results.shape}")
        # results = results.transpose()
        # print(f"transposed results: {results.shape}")
        # results = filter_detections(results, conf=conf)
        # print(f"After filtering: {results.shape}")

        # print(f"gauge Detected: {len(results)}")

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

def preprocess(image):
    image = image.astype(np.float32)
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, 0)
    # print(f"image type: {type(image)} {image.shape}")
    return image

def filter_detections(results, conf=0.5):
    # if model is trained on 1 class only
    if len(results[0]) == 5:
        # filter out the detections with confidence > thresh
        considerable_detections = [detection for detection in results if detection[4] > thresh]
        considerable_detections = np.array(considerable_detections)
        return considerable_detections

    # if model is trained on multiple classes
    else:
        A = []
        for detection in results:

            class_id = detection[4:].argmax()
            confidence_score = detection[4:].max()

            new_detection = np.append(detection[:4],[class_id,confidence_score])

            A.append(new_detection)

        A = np.array(A)

        # filter out the detections with confidence > thresh
        considerable_detections = [detection for detection in A if detection[-1] > thresh]
        considerable_detections = np.array(considerable_detections)

    return considerable_detections
