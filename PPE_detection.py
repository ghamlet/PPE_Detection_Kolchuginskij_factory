###UGMK top###
import cv2
import numpy as np
from ultralytics import YOLO


def load_models():
    model_person = YOLO("weights_and_cfg/yolov8n.pt")
    model_PPE = YOLO("weights_and_cfg/best.pt")

    return model_person, model_PPE


def processing(cap, model_person, model_PPE,fourcc):
    name = 0
    find = True

    while True:
        success, frame = cap.read()
        frame_copy = frame.copy()

        if not success:
            break
        
        #try to find a person
        res = model_person.track(frame, persist=False,  verbose = False, conf = 0.7, classes = 0, tracker="weights_and_cfg/bytetrack_custom.yaml")[0]
        boxes = res.boxes.xywh.cpu()
        
        # we can tracking a person
        if res.boxes.id:     
            track_ids = res.boxes.id.int().cpu().tolist()
            print("TRACK ID",track_ids)
            tracking = True
        else: tracking = False

        annotated_frame = res.plot()

        #if a person is being tracked 
        if tracking:
            if find:
                #we are recording this moment
                name+=1
                out = cv2.VideoWriter(f"detection_video/{str(name)}.avi", fourcc, 10.0, (1920,  1080))
                find = False

            #we use a model taken from Roboflow to detect personal protective equipment
            results = model_PPE.predict(source =frame_copy, conf = 0.5, classes = [0,1,2,3,4,7])[0]
            names = model_PPE.names #all classes of model
            objects_found = results.boxes.cls  #list numbers of classes
            for i in objects_found:
                print(names[int(i)])


            annotated_frame = results.plot()
            out.write(annotated_frame)
            cv2.waitKey(1)
        
        else:
            find = True

        cv2.imshow("YOLOv8 Tracking", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


    cap.release()
    out.release()
    cv2.destroyAllWindows()


def main():
    video_path = "kolchugino.mp4"
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')

    model_person, model_PPE = load_models()

    processing(cap, model_person, model_PPE, fourcc)
     


if __name__ == "__main__":     
    main()