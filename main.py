#1 Importing libraries
import cv2 as cv
from ultralytics import YOLO
from tqdm import tqdm
from utils import plate, preprocessing
import os
import time



#2 Loading the pretrain YOLO model
model = YOLO('./models/yolov8n.pt')



#3 Loading a video file
video_file = r"C:\Users\rezaa\spyder\Automated-Vehicle-Number-Plate-Recognition\sample_video\VID_20240715_130830.mp4"
cap = cv.VideoCapture(video_file)



#4 Getting frame properties
frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv.CAP_PROP_FPS)



#5 Since the cars are approaching from the top of the screen to the bootom so if we sample only those images of license plate
# which are nearer to the camera i.e., near the bottom of the screen we get better results
min_allowed_h = frame_height // 2.28
max_allowed_h = frame_height // 1.09


#6 Saving the video
folder_name = r'C:\Users\rezaa\spyder\Automated-Vehicle-Number-Plate-Recognition\output'
if not os.path.exists(folder_name):
    os.mkdir(folder_name)
output_file = os.path.basename(os.path.splitext(video_file)[0]) + '_' + str(time.time()) +'.mp4'
file_name = os.path.join(folder_name, output_file )
save_video = cv.VideoWriter(file_name, cv.VideoWriter_fourcc(*'mp4v'), 3, (frame_width,frame_height))




# 7 Variables to store car id and plate data
all_ids = []
plate_data = {}
c =0



# 8 Here we are looping through each frame and trying to find any cars
for i in tqdm(range(total_frames), desc="Processing", unit="iter", ncols=100):

    has_frame, frame = cap.read()
    
    if has_frame:
        frame_n = frame
        results = model.track(frame, persist=True, verbose = False)
        has_car = False
        

        # Looping through each box in a frame and checking if it's a car
        for box in results[0].boxes:
            if int(box.cls) in [2 , 3] and box.is_track :
                has_car = True
                


                # If a car is found we are getting it's unique id
                identity = int(box.id)
                x1,y1,x2,y2 = [int(num) for num in box.xyxy[0].tolist()] # getting the coodrinate of the box of the car
                frame_n = cv.rectangle(frame_n, (x1,y1), (x2,y2), (255,0,0) , 4) # Drawing a rectangle over the car
                has_plate = False



                # If the car has already been detected we are not further processing else we are trying to find the license
                # plate of the car
                with_in_limit = False
                if identity not in all_ids:
                    car_img = results[0].orig_img[y1:y2,x1:x2,:]
                    plate_img, has_plate, s1 = plate(car_img)
                    
                    
                    
                # If we have successfully found the license plate the next is we are trying to extract the text from the plate
                # and also keeping a record of the car whose plate has been identified so that in the next frame we only do the 
                # operations only for newer cars only
                    with_in_limit = min_allowed_h < s1 + y1 < max_allowed_h

                if has_plate and with_in_limit:
                    plate_text = preprocessing(plate_img)
                    if plate_text not in list(plate_data.values()):
                        plate_data[identity] = plate_text
                        all_ids.append(identity)
                


                # drawing text over the box to save it as a video file
                if identity in all_ids:
                    frame_n = cv.rectangle(frame_n, (x1,y1-25),(x2,y1), (0,0,0), -1)
                    frame_n = cv.putText(frame_n, plate_data[identity], (x1,y1),cv.FONT_HERSHEY_TRIPLEX ,1,(225,255,255),1, cv.LINE_AA)
                    
        save_video.write(frame_n)                                           
    else:
        break
print('Task Completed....')
cap.release()
save_video.release()
