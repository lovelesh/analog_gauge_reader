import cv2
import numpy as np
from gauge_detection.detection_inference import detection_gauge_face, find_center_bbox
from metadata import METER_CONFIG, get_gauge_details, save_metadata
from easygui import *

# egdemo()


# meter_index = 0

# METER_CONFIG = [{
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

# def get_gauge_details(box):
#     for index in range(len(METER_CONFIG)):
#         # print(f"index: {index}")
#         # point = tuple(METER_CONFIG[index]['center'])
#         # print(point)
#         if (METER_CONFIG[index]['center'][0] > int(box[0]) and 
#             METER_CONFIG[index]['center'][0] < int(box[2]) and
#             METER_CONFIG[index]['center'][1] > int(box[1]) and
#             METER_CONFIG[index]['center'][1] < int(box[3])):
#             return index
    
#     print("no Gauge details found")
#     return -1


# def find_center_bbox(box):
#     center = (int(box[0]) + (int(box[2] - box[0]) // 2), int(box[1]) + (int(box[3] - box[1]) // 2))
#     # print(f"center : {center}")
#     return center

# def save_metadata(x, y):
#     title = "Meter Metadata"
#     msg = "Enter metadata for the gauge"
#     fieldNames = ['name', 'start', 'end', 'unit']
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
#         # copy fieldValues to METER_CONFIG variable
#         METER_CONFIG.append({
#                 'name' : fieldValues[0],
#                 'start' : int(fieldValues[1]),
#                 'end' : int(fieldValues[2]),
#                 'unit' : fieldValues[3],
#                 'center' : (x, y) 
#         })
#         return True
    

def capture_xy(action, x, y, flags, *userdata):
    # global meter_index

    if action == cv2.EVENT_LBUTTONDBLCLK:
        # print(f"Meter Metadata: {METER_CONFIG[meter_index]}")
        if save_metadata(x, y) == True:
            print(f"Meter Metadata: {METER_CONFIG}")
            # meter_index += 1
        # except IndexError:
        #     print("adding empty dict entry")
        #     METER_CONFIG.append({
        #         'name' : '',
        #         'start' : 0,
        #         'end' : 0,
        #         'unit' : '',
        #         'center' : (0, 0)
        #     })
        #     print(METER_CONFIG)

def main():
    # Open the camera (0 is typically the default camera) 
    cap = cv2.VideoCapture(0) 
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1600)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    # cap.set(cv2.CAP_PROP_FOCUS, 20)
    # focus_value = 0  # Example focus value
    # zoom_value = 0 
    # global meter_index
    # Check if the camera opened successfully 
    while cap.isOpened():
        # Read a frame from video
        # print(f"Focus: {focus_value}")
        # print(f"zoom : {zoom_value}")
        # cap.set(cv2.CAP_PROP_ZOOM, zoom_value) 
        success, frame = cap.read()
        # make a copy of the image 
        temp = frame.copy()

        

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        box_color = (0, 255, 0)
        text_color = (0, 0, 255)
        box_thickness = 2
        text_thickness = 2

        # Show the frame 
        if success:
            image = np.asarray(frame)
            box, all_boxes = detection_gauge_face(image, model_path='models/gauge_detection_model_custom_trained.pt', conf=0.25)
            # for r in results:
            #     print(r.boxes)
            
            for index, box in enumerate(all_boxes):
                # print(f"x: {box[0]} y: {box[1]}, x: {box[2]}, y: {box[3]}")
                cv2.rectangle(frame, (int(box[0]), int(box[1])),
                            (int(box[2]), int(box[3])), box_color, box_thickness)
                
                # print(type(METER_CONFIG[index]['center'][0]))
                # print(METER_CONFIG[index]['center'])
                # x = tuple(METER_CONFIG[index]['center'])
                # print(f"Center : {find_center_bbox(box)}")
                try: 
                    gauge_index = get_gauge_details(box)
                    print(f"Gauge Details: {gauge_index}")
                    # print(f"Gauge details: {METER_CONFIG[get_gauge_details(box)]}")
                    cv2.putText(frame, f"#{index} {METER_CONFIG[gauge_index]['name']} {METER_CONFIG[gauge_index]['end']} {METER_CONFIG[gauge_index]['unit']}",
                                METER_CONFIG[gauge_index]['center'], font, font_scale, text_color, text_thickness)
                    
                except :
                    pass
                    # print("adding empty dict entry")
                    # METER_CONFIG.append({
                    #     'name' : '',
                    #     'start' : 0,
                    #     'end' : 0,
                    #     'unit' : '',
                    #     'center' : (0, 0)
                    # })
                    # print(METER_CONFIG)

            # Set callback for mouse events
            cv2.namedWindow("Webcam Feed")
            cv2.setMouseCallback("Webcam Feed", capture_xy)
            cv2.imshow("Webcam Feed", frame) 
            key = cv2.waitKey(0)  & 0xFF # Wait for a key press to close the window 
            if key == ord('r'):
                METER_CONFIG.clear()
                print("Metadata resetted")

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
    