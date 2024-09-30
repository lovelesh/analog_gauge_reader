from ultralytics import YOLO
from evaluation import constants
# import onnxruntime as ort
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common
from pycoral.adapters import detect
from PIL import Image
import time
import numpy as np

def   detection_gauge_face(img, model_path, conf=0.25):
    '''
    uses yolo v8 to get bounding box of gauge face
    :param img: numpy image
    :param model_path: path to yolov8 detection model
    :param conf: confidence score
    :return: highest confidence box for further processing and list of all boxes for visualization
    '''
    # task = "detect"
    # model = YOLO(model_path, task=task)  # load model
    # results = model.predict(img, conf=conf)     # , imgsz=constants.ORIGINAL_IMG_SIZE)
    # print(results[0])
    
    # Add mechanism to run edgetpu tflite format
    detection_interpreter = make_interpreter(model_path, device=':0') # Default coral edgeTPU
    detection_interpreter.allocate_tensors()
    input_details = detection_interpreter.get_input_details()
    # print(f"input_details: {input_details}")
    output_details = detection_interpreter.get_output_details()
    # print(f"output_details: {output_details}")
    input_index = detection_interpreter.get_input_details()[0]['index']
    # print(f"input_index: {input_index}")
    output_index = detection_interpreter.get_output_details()[0]['index']
    # print(f"output_index: {output_index}")

    # image = Image.fromarray(img)
    # print(image.size)
    # print(f"image size: {img.shape[:2]}")
    # input_tensor_size = common.input_tensor(detection_interpreter)
    # print(f"input tensor: {input_tensor_size.shape[:2]}")
    # input_size = (640, 640)
    # resize = (640, 640)
    # w, h = input_size
    # print(f" W: {w}, h: {h}")
    # common.set_input(detection_interpreter, img)
    # _, scale = common.set_resized_input(detection_interpreter, image.size, lambda size: image.resize(size, Image.LANCZOS))
    image = img.astype(np.int8)
    print(image.shape)
    image = np.reshape(image, (1, image.shape[1], image.shape[0], image.shape[2]))
    print(image.shape)
    start = time.perf_counter()
    detection_interpreter.set_tensor(input_index, image)
    detection_interpreter.invoke()
    # signature_list = detection_interpreter._get_full_signature_list()
    # print(signature_list)
    # scores = common.output_tensor(detection_interpreter, 0)[0]
    # print(f"scores: {scores}")
    # boxes = common.output_tensor(detection_interpreter, 1)[0]
    # print(f"boxes: {boxes}")
    # count = (int)(common.output_tensor(detection_interpreter, 2)[0])
    # print(f"count: {count}")
    # class_ids = common.output_tensor(detection_interpreter, 3)[0]
    # print(f"class ids: {class_ids}")
    results = detection_interpreter.get_tensor(output_index)
    print(f"output : {results.shape}")
    results = results[0].transpose()
    print(f"output : {results.shape}")
    results = filter_detections(results)
    print(f"No of detctions: {results.shape[0]}")
    results, confidences = rescale_back(results, 640, 640)
    print(f"after NMS: {len(results)}")
    # boxes = detect.get_objects(detection_interpreter, conf, 1)
    inference_time = time.perf_counter() - start
    print(f"inference time: {inference_time * 1000:.2f} ms")

    # if len(boxes) == 0:
    #     raise Exception("No gauge detected in image")

    # box_list = []
    # for box in boxes:
    #     box_list.append(box.xyxy[0].int())

    return results

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
    # print(len(results[0]))
    if len(results[0]) == 5:
        # print(f"results is 5")
        # filter out the detections with confidence > thresh
        considerable_detections = [detection for detection in results if detection[4] > conf]
        considerable_detections = np.array(considerable_detections)
        print(considerable_detections.shape)
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
        considerable_detections = [detection for detection in A if detection[-1] > conf]
        considerable_detections = np.array(considerable_detections)
        # print(considerable_detections)
    return considerable_detections

def non_max_suppression(boxes, conf_scores, iou_thresh=0.5):
    # boxes [[x1,y1, x2,y2], [x1,y1, x2,y2], ...]

    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    areas = (x2 - x1) * (y2 - y1)

    order = conf_scores.argsort()

    keep = []
    keep_confidences = []

    while len(order) > 0:
        idx = order[-1]
        A = boxes[idx]
        conf = conf_scores[idx]

        order = order[:-1]

        xx1 = np.take(x1, indices=order)
        yy1 = np.take(y1, indices=order)
        xx2 = np.take(x2, indices=order)
        yy2 = np.take(y2, indices=order)

        keep.append(A)
        keep_confidences.append(conf)

        # iou = inter / union

        xx1 = np.maximum(x1[idx], xx1)
        yy1 = np.maximum(y1[idx], yy1)
        xx2 = np.minimum(x2[idx], xx2)
        yy2 = np.minimum(y2[idx], yy2)

        w = np.maximum(xx2 - xx1, 0)
        h = np.maximum(yy2 - yy1, 0)

        intersection = w * h

        # union = Area + other_area - intersection
        other_area = np.take(areas, indices=order)
        union = areas[idx] + other_area - intersection

        iou = intersection / union

        boleans = iou <iou_thresh

        order = order[boleans]

        # order = [2,0,1]  boleans = [True, False, True]
        # order = [2,1]

    return keep, keep_confidences

def rescale_back(results,img_w,img_h):
    # print(f"rescaled : {results.shape}")
    cx, cy, w, h, class_id, confidence = results[:,0], results[:,1], results[:,2], results[:,3], results[:,4], results[:,-1]
    # convert x1,y1, x2,y2 into uint8 format
    print(f"cx:{cx}, cy:{cy}, w:{w}, h:{h} {cx.dtype} conf: {confidence}")
    # cx = cx.astype(np.uint16)
    # cy = cy.astype(np.uint16)
    # w = w.astype(np.uint16)
    # h = h.astype(np.uint16) 
   
    # print(f"cx {type(cx)}")
    cx = cx/640.0 * img_w
    cy = cy/640.0 * img_h
    w = w/640.0 * img_w
    h = h/640.0 * img_h
    x1 = cx - w/2
    y1 = cy - h/2
    x2 = cx + w/2
    y2 = cy + h/2

    print(f"x1:{x1}, y1:{y1}, x2:{x2}, y2:{y2}")

    boxes = np.column_stack((x1, y1, x2, y2, class_id))
    # print(boxes)
    keep, keep_confidences = non_max_suppression(boxes,confidence)
    # print(np.array(keep).shape)
    return keep, keep_confidences

