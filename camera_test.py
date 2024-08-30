import cv2 
 
# Open the camera (0 is typically the default camera) 
cap = cv2.VideoCapture(0) 
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1600)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
# cap.set(cv2.CAP_PROP_FOCUS, 20)
focus_value = 0  # Example focus value
zoom_value = 0 
# Check if the camera opened successfully 
while cap.isOpened():
    # Read a frame from video
    # print(f"Focus: {focus_value}")
    print(f"zoom : {zoom_value}")
    cap.set(cv2.CAP_PROP_ZOOM, zoom_value) 
    success, frame = cap.read()

    # Show the frame 
    if success: 
        cv2.imshow("Webcam Feed", frame) 
        cv2.waitKey(0)  # Wait for a key press to close the window 
    else: 
        print("Error: Could not read frame.") 
    
    # focus_value += 1
    zoom_value += 1
 
# Release the camera and close windows 
cap.release() 
cv2.destroyAllWindows() 