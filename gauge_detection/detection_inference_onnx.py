# from ultralytics import YOLO
from evaluation import constants
# import onnxruntime as ort
import numpy as np
import cv2

def detection_gauge_face(img, model_path='yolov8n.pt', conf=0.25):
    '''
    uses yolo v8 to get bounding box of gauge face
    :param img: numpy image
    :param model_path: path to yolov8 detection model
    :return: highest confidence box for further processing and list of all boxes for visualization
    '''
 
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
    img, img_height, img_width = preprocess(img)
    
    # load the model
    net = cv2.dnn.readNetFromONNX(model_path)

    # feed the image
    net.setInput(img)

    # run the inferecence
    out = net.forward()
    # print(f"inference shape: {out.shape}")    # should be (1, 5, 8400)
    
    results = out[0]
    results = results.transpose()
    # print(f"output transposed: {results.shape}")

    # filter outputs based on confidence
    results = filter_detections(results, conf)
    # print(f"filterred: {results.shape}")

    rescaled_results, confidences = rescale_back(results, img_width, img_height)

    return rescaled_results

def find_center_bbox(box):
    center = (int(box[0]) + (int(box[2] - box[0]) // 2), int(box[1]) + (int(box[3] - box[1]) // 2))
    return center

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_height, img_width = img.shape[:2]
    # print(img.shape)
    # resize the image to (640, 640) as the default model runs on this resolution
    img = cv2.resize(img, (640, 640))
    # transpose the image from (640, 640, 3) to (3, 640, 640) 
    img = img.transpose(2, 0, 1)
    # print(img.shape)

    # reshape the image to (1, 3, 640 ,640)
    img = img.reshape(1, 3, 640, 640)
    # set scale 0-1
    img = img/255.0
    return img, img_height, img_width

def filter_detections(results, conf=0.5):
    # if model is trained on 1 class only
    if len(results[0]) == 5:
        # print("result is 5")
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

    return considerable_detections

def NMS(boxes, conf_scores, iou_thresh = 0.55):

    #  boxes [[x1,y1, x2,y2], [x1,y1, x2,y2], ...]

    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    areas = (x2-x1)*(y2-y1)

    order = conf_scores.argsort()

    keep = []
    keep_confidences = []

    while len(order) > 0:
        idx = order[-1]
        A = boxes[idx]
        conf = conf_scores[idx]

        order = order[:-1]

        xx1 = np.take(x1, indices= order)
        yy1 = np.take(y1, indices= order)
        xx2 = np.take(x2, indices= order)
        yy2 = np.take(y2, indices= order)

        keep.append(A)
        keep_confidences.append(conf)

        # iou = inter/union

        xx1 = np.maximum(x1[idx], xx1)
        yy1 = np.maximum(y1[idx], yy1)
        xx2 = np.minimum(x2[idx], xx2)
        yy2 = np.minimum(y2[idx], yy2)

        w = np.maximum(xx2-xx1, 0)
        h = np.maximum(yy2-yy1, 0)

        intersection = w*h

        # union = areaA + other_areas - intesection
        other_areas = np.take(areas, indices= order)
        union = areas[idx] + other_areas - intersection

        iou = intersection/union

        boleans = iou < iou_thresh

        order = order[boleans]

        # order = [2,0,1]  boleans = [True, False, True]
        # order = [2,1]
    return keep, keep_confidences

def rescale_back(results,img_w,img_h):
    cx, cy, w, h, class_id, confidence = results[:,0], results[:,1], results[:,2], results[:,3], results[:,4], results[:,-1]
    cx = cx/640.0 * img_w
    cy = cy/640.0 * img_h
    w = w/640.0 * img_w
    h = h/640.0 * img_h
    x1 = cx - w/2
    y1 = cy - h/2
    x2 = cx + w/2
    y2 = cy + h/2

    boxes = np.column_stack((x1, y1, x2, y2, class_id))
    keep, keep_confidences = NMS(boxes,confidence)
    # print(np.array(keep).shape)
    return keep, keep_confidences

