from ultralytics import YOLO
import os


def train_yolo_model(task, data_file, model):

    training_epochs = 20
    plots = True
    conf = 0.5  # confidence threshold yolo
    export_format = "onnx"
    # imgsz = 1024

    model.train(task=task,
                data=data_file,
                plots=plots,
                epochs=training_epochs,
                conf=conf)
                # imgsz=imgsz)
    metrics = model.val()
    print(metrics.box.map)  # mAP50-95
    print(metrics.box.map50)  # mAP50
    print(metrics.box.map75)  # mAP75
    print(metrics.box.maps)  # list of mAP50-95 for each category
    
    model.export(format=export_format)

def predict_yolo_model(task, source_file_path, model):

    image_size = 1024
    visualize = False
    show = False
    save = True
    conf = 0.75
    
    model.predict(task=task,
                  source=source_file_path,
                #    imgsz=image_size,
                   conf=conf,
                   visualize=visualize,
                   show=show,
                   save=save)

if __name__ == "__main__":
    print(os.getcwd())
    ''' Start the training for detection model'''
    # detection_data_file = "data/detection/data.yaml"
    # detection_model_name = "yolov8n.pt"
    # detection_model = YOLO(detection_model_name)

    # detection_inference_model_file_path = "runs/detect/train12/weights/best.onnx"
    # detection_inference_data_path = "data/detection/test/images"

    # Train the detection model
    # train_yolo_model('detect', detection_data_file, detection_model)

    # test the detection model
    # predict_yolo_model('detect', inference_data_path, detection_model)

    # with open(detection_data_file, 'r') as f:
    #     print(f.read())
    
    '''Start training for segmentation model'''
    # segmentation_data_file = "data/segmentation/data.yaml"
    # segmentation_model_name = "yolov8n-seg.pt"
    # segmentation_model = YOLO(segmentation_model_name)
    
    segmentation_inference_model_file_path = "runs/segment/train2/weights/best.onnx"
    segmentation_inference_data_path = "data/segmentation/test/images"
    segmentation_model = YOLO(segmentation_inference_model_file_path)

    # Train the detection model
    # train_yolo_model('segment', segmentation_data_file, segmentation_model)

    # test the detection model
    predict_yolo_model('segment', segmentation_inference_data_path, segmentation_model)