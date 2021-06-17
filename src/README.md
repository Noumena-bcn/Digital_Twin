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

The JSON format that is created for occupancy is the following one:

```
	{
	    "Zone": {
		           "Zone 1": (valor)
		           "Zone 2": (valor)
		           "Zone 3": (valor) 
		           "Zone 4": (valor)
		           "Zone 5": (valor)     
		         }
	  "General": (valor)
	}
```
The result it can be found on [JSON_Occupancy](https://github.com/Noumena-bcn/Digital_Twin/tree/main/src/JSON_Occupancy)

The JSON format that is created for predictive modelling is the following one:

```
	{
		"Zone": zone,
		"class_title": "person",
		"cam":{
		    "camX":{
		            "coordinates":[point_box], "width": width, "height": height
		            }
			},
		"coordinates_map": [(point)],
		"frame_number": frame_num,
    	}
```
The result it can be found on [JSON_Output](https://github.com/Noumena-bcn/Digital_Twin/tree/main/src/JSON_Output)

## digital_twin_func

Use ```digital_twin_func.py```.

In this code, the image and the original points are transformed to a top view image.
Depending on the input camera, the result will apear on the predefined zone.


```
usage: digital_twin.py [-h] [-iv INPUT_VIDEO_PATH] [-id INPUT_DETECTIONS_PATH]
                       [-o OUTPUT_PATH] [-cl CLASSES] [-ds] [-v]
optional arguments:
  -h, --help            show this help message and exit
  -iv INPUT_VIDEO_PATH, --input_video_path INPUT_VIDEO_PATH
                        Path to the video.
  -id INPUT_DETECTIONS_PATH, --input_detections_path INPUT_DETECTIONS_PATH
                        Path to the detections file. E.g.
                        '..\..\Input_Videos\GEYE0010_THM.mp4'. Default is 0
                        (webcam).
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        Path to the video output.
  -cl CLASSES, --classes CLASSES
                        Classes of interest. Default: all COCO classes. Eg.
                        'person;cell phone;car' (without blanks between
                        classes).
  -ds, --dont_show      Don't show video output.
  -v, --verbose         Show detailed info of tracked objects.

```

Also the results are on that folder: [Floor videos](https://drive.google.com/drive/folders/14tlNH5MsLtoytXrMXISb9HL-4G3_EfgS)
