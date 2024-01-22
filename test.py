###UGMK top###
import cv2
import numpy as np
from ultralytics import YOLO



model_person = YOLO("weights_and_cfg/yolov8n.pt")
model_PPE = YOLO("weights_and_cfg/best.pt")



video_path = "kolchugino.mp4"
cap = cv2.VideoCapture(video_path)  #self._name = name + '.mp4'

fourcc = cv2.VideoWriter_fourcc(*'MP4V')

count = 0
find = True

while cap.isOpened():
    success, frame = cap.read()
    

    if not success:
        break

    frame_copy = frame.copy()
    
    #try to find a person
    res = model_person.track(frame, persist=False,  verbose = False, conf = 0.7, classes = 0, tracker="weights_and_cfg/bytetrack_custom.yaml")[0]

    
    # we can tracking a person
    if res.boxes.id:     
        track_ids = res.boxes.id.int().cpu().tolist()
        #print("TRACK ID",track_ids)
        tracking = True
    else: tracking = False

    annotated_frame = res.plot()

    #if a person is being tracked 
    if tracking:
        if find:
            #we are recording this moment
            count+=1

            frame_width = int(cap.get(3)) 
            frame_height = int(cap.get(4)) 
            size = (frame_width, frame_height)  #set the image size only this way   

            out = cv2.VideoWriter(f"output_{count}.mp4", fourcc, 30.0, size)
            find = False #create a new object of the video class when tracking has stopped

        #we use a model taken from Roboflow to detect personal protective equipment
        results = model_PPE.predict(source =frame_copy, verbose = False, conf = 0.5, classes = [0,1,2,3,4,7])[0]
        
        
        annotated = results.plot()
        out.write(annotated_frame)
        cv2.waitKey(1)
    
    else:
        find = True #when there is no one in the frame, tracking becomes false and find is true
                    #so that when a person reappears, a new recording begins

    cv2.imshow("Tracking", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

out.release()
cap.release()

cv2.destroyAllWindows()







 
     

