# Multiprocess, Homography and MongoDB

This directory contains the scripts for the multiprocess codes, the homgraphy transformation and the mongoDB storage 

## Requirements

- OpenCV built with CUDA GPU support
- [requirements.txt](requirements.txt)

Create a python virtual environement.
Configure python3 as the default python interpreter for this venv.
Link the OpenCV installation (cv2.so) to this venv.
Then 
```
pip install --upgrade pip
pip install requirements.txt 
```

## multiprocess_common

Use ```multiprocess_common.py```.

This is the main code. Running that you'll be able to execute all the codes that are implemented on it at the same time.

It requires of:

```
import digital_twin_func as dt
import mongo as stm
```
It also create a floor video, where you can watch the flow of the detected people on the differents zones of the building

On that folder, you'll find the visualizacion results: [Floor videos](https://drive.google.com/drive/folders/14tlNH5MsLtoytXrMXISb9HL-4G3_EfgS)

## digital_twin_func

Use ```digital_twin_func.py```.

In this code, the image and the original points are transformed to a top view image.
Depending on the input camera, the result will apear on the predefined.





