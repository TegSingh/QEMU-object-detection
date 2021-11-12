import cv2
import numpy as np
import time


def main():
    # capture frames from a video
    cap = cv2.VideoCapture('Video.mp4')
    
    # Trained XML classifiers describes some features of some object we want to detect
    car_cascade = cv2.CascadeClassifier('cars.xml')
    bikes_cascade = cv2.CascadeClassifier('bikes.xml')
    pedestrian_cascade = cv2.CascadeClassifier('pedestrian.xml')
    bus_cascade = cv2.CascadeClassifier('bus.xml')


    print("Generated classification objects through online XML files")
    i = 0
    
    # Create counter variables
    car_count = 0
    bike_count = 0
    pedestrian_count = 0
    bus_count = 0

    num_frames = 0
    
    # Get the first 500 inference times
    while i < 500:
        
        ret, frames = cap.read()
        num_frames += len(frames)

        gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
        
        # Generate Inference timer
        start = time.time()

        # Detects vehicles of different sizes in the input image
        cars = car_cascade.detectMultiScale(gray, 1.1, 1)
        bikes = bikes_cascade.detectMultiScale(gray, 1.1, 1)
        pedestrian = pedestrian_cascade.detectMultiScale(gray, 1.1, 1)
        bus = bus_cascade.detectMultiScale(gray, 1.1, 1)

        # Finish Inference timing        
        end = time.time()
        
        print("Inference time: ", end - start)
        
        # Rectangle around cars
        
        for (x,y,w,h) in cars:
            cv2.rectangle(frames,(x,y),(x+w,y+h),(0,0,255),2)
            print("Boundary box: car: ", x, " ", y, " ",  w, " ", h)
        car_count += len(cars)

        # Rectangle around Bikes
        for (x,y,w,h) in bikes:
            cv2.rectangle(frames,(x,y),(x+w,y+h),(0,255,0),2)
            print("Boundary box: bikes: ", x, " ", y, " ",  w, " ", h)

        bike_count += len(bikes)

        # Rectangle around Pedestrian
        for (x,y,w,h) in pedestrian:
            cv2.rectangle(frames,(x,y),(x+w,y+h),(255,0,0),2)
            print("Boundary box: pedestrian: ", x, " ", y, " ",  w, " ", h)
        
        pedestrian_count += len(pedestrian)

        # Rectangle around Bus
        for (x,y,w,h) in bus:
            cv2.rectangle(frames,(x,y),(x+w,y+h),(255,255,255),2)
            print("Boundary box: bus: ", x, " ", y, " ",  w, " ", h)

        bus_count += len(bus)

        i += 1

        # Display frames in a window 
        # cv2.imshow('video2', frames)
    
        # Wait for Esc key to stop
        if cv2.waitKey(33) == 27:
            break

    # De-allocate any associated memory usage
    cv2.destroyAllWindows()

    print("Total inferences: ", i)
    print("Cars: ", car_count, " Bikes: ", bike_count, " Pedestrian: ", pedestrian_count, " Bus: ", bus_count)
    print("Number of frames: ", num_frames)

if __name__ == "__main__":
    main()
