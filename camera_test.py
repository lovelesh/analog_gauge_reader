import cv2
import numpy as np
from gauge_detection.detection_inference import detection_gauge_face, find_center_bbox
import metadata
from easygui import *
import json

# egdemo()

WINDOW_NAME = "Webcam Feed"

# meter_index = 0

# meter_config = [{
#             'name' : 'boiler pressure',
#             'start_value' : 0,
#             'range' : 800,
#             'unit' : 'kPa',
#             'center' : (864, 660)
# }, {
#             'name' : 'tank pressure',
#             'start_value' : 0,
#             'range' : 200,
#             'unit' : 'psi',
#             'center' : (870, 394)
# }, {
#             'name' : 'reservoir pressure',
#             'start_value' : 0,
#             'range' : 400,
#             'unit' : 'psi',
#             'center' : (504, 610)
# }]

# meter_config = []

# def get_gauge_details(box):
#     print(type(meter_config))
#     for index in range(len(meter_config)):
#         print(meter_config[index]['center'][0])
#         if (meter_config[index]['center'][0] > int(box[0]) and 
#             meter_config[index]['center'][0] < int(box[2]) and
#             meter_config[index]['center'][1] > int(box[1]) and
#             meter_config[index]['center'][1] < int(box[3])):
#             return index
    
#     print("no Gauge details found")
#     raise Exception("no Gauge details found")

# def save_metadata(x, y):
#     title = "Meter Metadata"
#     msg = "Enter metadata for the gauge"
#     fieldNames = ["Name", "Start", "End", "Unit"]
#     fieldValues = []
#     fieldValues = multenterbox(msg, title, fieldNames)
#     print(f"received info: {fieldValues}")
#     if fieldValues == None:
#         return False
#     errmsg = ""
#     for i in range(len(fieldNames)):
#         if fieldValues[i].strip == "" :
#             errmsg = errmsg + ("%s is required\n" % fieldNames[i])
#     if errmsg == "":
#         # copy fieldValues to meter_config variable
#         meter_config.append({
#                 "name" : fieldValues[0],
#                 "start" : int(fieldValues[1]),
#                 "end" : int(fieldValues[2]),
#                 "unit" : fieldValues[3],
#                 "center" : list((x, y)) 
#         })
#         print(meter_config)
#         return True
      
    

def capture_xy(action, x, y, flags, *userdata):
    # global meter_index

    if action == cv2.EVENT_LBUTTONDBLCLK:
        # print(f"Meter Metadata: {meter_config[meter_index]}")
        metadata.save_metadata(x, y)
        # write_json_file(userdata[0], meter_config)


def write_json_file(filename, dictionary):
    file_json = json.dumps(dictionary, indent=4)
    with open(filename, "w") as outfile:
        outfile.write(file_json)
    outfile.close()

def read_json_file(filename):
    dictionary = []
    with open(filename, "r") as infile:
        dictionary = json.load(infile)
    infile.close()
    return dictionary


def main():
    # Open the camera (0 is typically the default camera) 
    cap = cv2.VideoCapture(2) 
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1600)
    # cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    # cap.set(cv2.CAP_PROP_FOCUS, 20)
    # focus_value = 0  # Example focus value
    # zoom_value = 0 
    metadata_path = "metadata.json"
    metadata.meter_config = read_json_file(metadata_path)
    print(f"Read METER Data: {metadata.meter_config}")

    # Check if the camera opened successfully 
    while cap.isOpened():
        # Read a frame from video
        # print(f"Focus: {focus_value}")
        # print(f"zoom : {zoom_value}")
        # cap.set(cv2.CAP_PROP_ZOOM, zoom_value) 
        success, frame = cap.read()        

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        box_color = (0, 255, 0)
        text_color = (0, 0, 255)
        box_thickness = 2
        text_thickness = 2

        # Show the frame 
        if success:
            image = np.asarray(frame)
            all_boxes = detection_gauge_face(image, model_path='models/gauge_detection_model_custom_trained_saved_model_640/gauge_detection_model_custom_trained_full_integer_quant_edgetpu.tflite', conf=0.25)
            # for r in results:
            #     print(r.boxes)
            
            for index, box in enumerate(all_boxes):
                # print(f"x: {box[0]} y: {box[1]}, x: {box[2]}, y: {box[3]}")
                cv2.rectangle(frame, (int(box[0]), int(box[1])),
                            (int(box[2]), int(box[3])), box_color, box_thickness)
                
                # print(type(meter_config[index]['center'][0]))
                # print(meter_config[index]['center'])
                # x = tuple(meter_config[index]['center'])
                # print(f"Center : {find_center_bbox(box)}")
                try: 
                    gauge_index = metadata.get_gauge_details(box)
                    print(f"Gauge Details: {gauge_index} {metadata.meter_config[gauge_index]}")
                    cv2.putText(frame, f"#{index} #{metadata.meter_config[gauge_index]['id']} {metadata.meter_config[gauge_index]['name']} {metadata.meter_config[gauge_index]['end']} {metadata.meter_config[gauge_index]['unit']}",
                                find_center_bbox(box), font, font_scale, text_color, text_thickness)
                    
                except :
                    pass

            # Set callback for mouse events
            cv2.namedWindow(WINDOW_NAME)
            # cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.setMouseCallback(WINDOW_NAME, capture_xy)
            cv2.imshow(WINDOW_NAME, frame) 
            key = cv2.waitKey(0)  & 0xFF # Wait for a key press to close the window 
            if key == ord('r'):
                metadata.meter_config.clear()
                print("Metadata resetted")
                print(metadata.meter_config)

            elif key == ord('d'):
                print(f"Current METER Data: {metadata.meter_config}")
            
            elif key == ord('s'):
                print(f"Saving METER Data: {metadata.meter_config}")
                write_json_file(metadata_path, metadata.meter_config)

            elif key == ord('o'):
                metadata.meter_config = read_json_file(metadata_path)
                print(f"Read METER Data: {metadata.meter_config}")
            
            elif key == ord('q'):
                break
        else: 
            print("Error: Could not read frame.") 
        
        # focus_value += 1
        # zoom_value += 1
    
    # Release the camera and close windows 
    cap.release() 
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
