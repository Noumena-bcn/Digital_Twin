# Digital_Twin

New repository for the Noumena Robotics Digital Twin.

Each directory is briefly described hereafter, and more detailed README files can be found in each of them.

In addition, a guide for building OpenCV with CUDA support for the DNN module is maintained [here](https://docs.google.com/document/d/1qRDi8voHjlAEyq6xXm3BSpu1nlvRAQ3B0Ip0NqSYwVY/edit?usp=sharing).


## Map

Visualization of the focus of the cameras and divided in different zones according to the location of the cameras 

Maps:

- Camera location
- Map zones 

## Cams

Inside that folder, you can find the focus of each camera

## yolov4-opencv

General purpose scripts to use [YOLOv4](https://github.com/AlexeyAB/darknet) with OpenCV's DNN module for object detection.

The output can be stored as a video or as JSON files.

![image](https://user-images.githubusercontent.com/62296738/115378528-edcf7e00-a1d0-11eb-80b1-7b28a3eff255.png)

## src

This is the main folder of the project, it contains the codes that were used in this project after the detection was runned. Techniques such as multiprocessing and homography have been used to achieve this task.

Topics:

- Multiprocess
- Homography 
- Mongo

### Multiprocess

The multiprocessing module allows the programmer to fully leverage multiple processors on a given machine. The API used is similar to the classic threading module. It offers both local and remote concurrency.

Is used to execute the codes for each camera at the same time. In this case, the main code communicate with the multiprocess codes to extract the necessary data to plot the positions on the map.

### Homography

Briefly, the planar homography relates the transformation between two planes.

![image](https://user-images.githubusercontent.com/62296738/122360320-9b070e00-cf56-11eb-8188-d45a5d2274b5.png)

The procedure is shown below

### Mongo

MongoDB is an open source NoSQL database management program. NoSQL is used as an alternative to traditional relational databases.

It save the necessary information in JSON files format.

How to install:
- MongoDB: https://docs.mongodb.com/manual/tutorial/install-mongodb-on-ubuntu/
- MongoDB Compass:https://docs.mongodb.com/compass/current/install/


## Initial videos

The initial videos are on the following folder:

- [Initial videos](https://drive.google.com/drive/u/0/folders/1Dn62ek8EV1BpPnlsOO2rYxZ-wQqDSRWH)



