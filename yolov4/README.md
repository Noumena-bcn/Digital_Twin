# People detection with Yolov4 and Opencv

This directory contains the scripts for the detections of the people on the different cameras located on La Salle 

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
## yolov4_opencv_json

On that code, after incuding the necessaries parameters to be executed shown bellow, it create a JSON file for all the detections on the video.

```
usage: yolov4_opencv_json.py [-h] [-i INPUT_PATH] [-cl CLASSES] [-w WEIGHTS]
                             [-cfg CONFIG] [-ct CONF_THRESH] [-nt NMS_THRESH]
                             [-t] [--cpu]

Object Detection Algorithm Evaluation

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_PATH, --input_path INPUT_PATH
                        Path to the video on which detection must be
                        performed. E.g. '..\..\Input_Videos\LaSallevid.mp4'.
                        Default is 0 (webcam).
  -cl CLASSES, --classes CLASSES
                        Classes of interest. Default: all COCO classes. Eg.
                        'person;cell phone;car' (without blanks between
                        classes).
  -w WEIGHTS, --weights WEIGHTS
                        Path to the model weights file. Default is yolo-coco-
                        data\yolov4.weights
  -cfg CONFIG, --config CONFIG
                        Path to the model config file. Default is yolo-coco-
                        data\yolov4-RESOLUTION.cfg
  -ct CONF_THRESH, --conf_thresh CONF_THRESH
                        Confidence threshold. Default is 0.15.
  -nt NMS_THRESH, --nms_thresh NMS_THRESH
                        Non Maxima Suppression Threshold. Default is 0.25.
  -t, --track           Enable tracking with deepsort.
  --cpu                 Use CPU instead of GPU for detection. By default the
                        CUDA GPU backend is used.

```

The final structrue of the JSON file is the next one:

```
{
	"video_source": video_name,
	"detection_model": model_name,
	"frame_number": frame_number,
	"class_title": class_name,
	"bbox": [coordinates of the box],
	"coordinates_image": {"x": x_obj, "y": y_obj}
}
```

If it also implements the tracker, the result it'll be:

```
{
	"video_source": video_name,
	"detection_model": model_name,
	"frame_number": frame_number,
	"class_title": class_name,
	"bbox": [coordinates of the box],
	"coordinates_image": {"x": x_obj, "y": y_obj}
	"track_id": track_id
}
```

## yolov4_opencv_json

It create a video drawing a bounding box for each detection on the frame.

```
-h, --help            show this help message and exit
  -i INPUT_PATH, --input_path INPUT_PATH
                        Path to the video on which detection must be
                        performed. Default is 0 for the computer's webcam.
  -cl CLASSES, --classes CLASSES
                        Classes of interest. Default: all COCO classes. Eg.
                        'person;cell phone;car' (without blanks between
                        classes).
  -s, --save_video      Save video output to mp4 file.
  -o OUT_PATH, --out_path OUT_PATH
                        Path for the output video file. Default is
                        'output/yolov4-output/
  -w WEIGHTS, --weights WEIGHTS
                        Path to the model weights file. Default is yolo-coco-
                        data\yolov4.weights
  -cfg CONFIG, --config CONFIG
                        Path to the model config file. Default is yolo-coco-
                        data\yolov4-RESOLUTION.cfg
  -ct CONF_THRESH, --conf_thresh CONF_THRESH
                        Confidence threshold. Default is 0.15.
  -nt NMS_THRESH, --nms_thresh NMS_THRESH
                        Non Maxima Suppression Threshold. Default is 0.25.
  -t, --track           Enable tracking with deepsort.
  --cpu                 Use CPU instead of GPU for detection. By default the
                        CUDA GPU backend is used.
  -sh, --show           Show video output.
```

All the detected videos are saved on the folder of [Noumena](https://drive.google.com/drive/u/0/folders/1M0l9dxL4aROsXvTc61jrJhNlM_yu1Zqr)
