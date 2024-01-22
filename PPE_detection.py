import cv2
import numpy as np
from ultralytics import YOLO
import statistical_graph as stat

def TelegramBot():
    pass


def load_models():
    model_person = YOLO("weights_and_cfg/yolov8n.pt")
    model_PPE = YOLO("weights_and_cfg/best.pt")

    return model_person, model_PPE


def processing(cap: cv2.VideoCapture, model_person: YOLO, model_PPE: YOLO):
    count = 0
    find = True
    ppe_person = []

    while True:
        success, frame = cap.read()
        frame_copy = frame.copy()

        if not success:
            break
        
        #try to find a person
        res = model_person.track(frame, persist=False,  verbose = False, conf = 0.7, classes = 0, tracker="weights_and_cfg/bytetrack_custom.yaml")[0]
        
        # we can tracking a person
        if res.boxes.id:     
            track_ids = res.boxes.id.int().cpu().tolist()
            #print("Track Id",track_ids)
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

                out = cv2.VideoWriter(f"detection_video/output_{str(count)}.mp4", cv2.VideoWriter_fourcc(*'MP4V'), 30.0, size)
                find = False #create a new object of the video class when tracking has stopped

            #we use a model taken from Roboflow to detect personal protective equipment
            results = model_PPE.predict(source =frame_copy, conf = 0.5, verbose = False, classes = [0,1,2,3,4,7])[0]
            
            names = model_PPE.names 
            #{0: 'Hardhat', 1: 'Mask', 2: 'NO-Hardhat', 3: 'NO-Mask', 4: 'NO-Safety Vest', 5: 'Person', 6: 'Safety Cone', 7: 'Safety Vest', 8: 'machinery', 9: 'vehicle'}
            
            objects_found = results.boxes.cls  #list numbers of classes
            for num in objects_found:
                object = names[int(num)]
                if object not in ppe_person: #While tracking is working for one person, we collect information about what he is wearing
                    ppe_person.append(object)
            

            annotated_frame = results.plot()
            out.write(annotated_frame)  #adding a frame to a stream
            cv2.waitKey(1)
        
        else:
            find = True #when there is no one in the frame, tracking becomes false and find is true
                        #so that when a person reappears, a new recording begins


            if len(ppe_person) != 0: #if tracking has ended and there is information about the clothes of the previous person
                stat.write_excel(ppe_person)  #write information into a table
            ppe_person = []

        cv2.imshow("Tracking", annotated_frame)

        k = cv2.waitKey(1) 
        if k == ord("q"):
            break


    cap.release()
    out.release()
    cv2.destroyAllWindows()


def main():
    video_path = "kolchugino.mp4"
    cap = cv2.VideoCapture(video_path)

    model_person, model_PPE = load_models()

    processing(cap, model_person, model_PPE)
     

if __name__ == "__main__":     
    main()