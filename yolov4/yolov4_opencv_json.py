import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import time
import argparse
import numpy as np
import json

import progressbar as pb

def store_bson(objects, file_name):
    """
    Stores dict objects in a JSON file.
    Objects can be either one dict object or a list of dict objects.
    """
    with open(file_name, 'w') as f:
        json.dump(objects, f)
        f.close()

def parser():
    parser = argparse.ArgumentParser(description="Object Detection Algorithm Evaluation")
    parser.add_argument("-i", "--input_path", type=str, default='',
                        help="Path to the video on which detection must be performed. E.g. '..\\..\\Input_Videos\\GEYE0010_THM.mp4'. Default is 0 (webcam).")
    parser.add_argument("-cl", "--classes", type=str, default='',
                        help="Classes of interest. Default: all COCO classes. Eg. 'person;cell phone;car' (without blanks between classes).")
    parser.add_argument("-w", "--weights", type=str, default=os.path.join("yolo-coco-data","yolov4.weights"),
                        help="Path to the model weights file. Default is yolo-coco-data\yolov4.weights")
    parser.add_argument("-cfg", "--config", type=str, default="",
                        help="Path to the model config file. Default is yolo-coco-data\yolov4-RESOLUTION.cfg")
    #parser.add_argument("-r", "--resolution", type=int, default=416,
    #                    help="Resolution of the network input layer. Default is 416.")
    parser.add_argument("-ct", "--conf_thresh", type=float, default=0.15,
                        help="Confidence threshold. Default is 0.15.")
    parser.add_argument("-nt", "--nms_thresh", type=float, default=0.25,
                        help="Non Maxima Suppression Threshold. Default is 0.25.")
    parser.add_argument("-t", "--track", action='store_true',
                        help="Enable tracking with deepsort.") 
    parser.add_argument("--cpu", action='store_true',
                        help="Use CPU instead of GPU for detection. By default the CUDA GPU backend is used.") 
    return parser.parse_args()

#def check_args(args):
#    if not path.exists(args.gt_path):
#        raise(ValueError("Invalid ground truth annotations path {}".format(path.abspath(args.gt_path))))
#    if not path.exists(args.gt_classes):
#        raise(ValueError("Invalid ground truth classes path {}".format(path.abspath(args.gt_classes))))

args = parser()
#check_args(args)
#if args.img_path[-1] not in "/\\":
#    args.img_path = args.img_path + '/'
#if args.gt_path[-1] not in "/\\":
#    args.gt_path = args.gt_path + '/'

TRACK = args.track
if TRACK:
    # deep sort imports
    from deep_sort import preprocessing, nn_matching
    from deep_sort.detection import Detection
    from deep_sort.tracker import Tracker
    from tools import generate_detections as gdet
        # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None

    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    try:
        encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    except Exception as e:
        print(e)
        print()
        print("Make sure {} exists.".format(os.path.abspath(model_filename)))
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

# Initialisation of the network

weight_file = args.weights
cfg_file = args.config if args.config else os.path.join("yolo-coco-data","yolov4.cfg")

print()
print("Confidence threshold: " + str(args.conf_thresh))
print("NMS threshold: " + str(args.nms_thresh))
#print("Input layer resolution: " + str(args.resolution))
print("Use CPU") if args.cpu else print("Use GPU")
print()

RESOLUTION = 640
CONFIDENCE_THRESHOLD = 0.15
NMS_THRESHOLD = 0.25
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

class_names = []
with open(os.path.join("yolo-coco-data","coco.names"), "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

if args.classes:
    interesting_classes = args.classes.split(';')
else:
    interesting_classes = ['dog','car', 'bicycle', 'skateboard', 'bus', 'person', 'motorbike', 'scooter', 'truck']

input_path = args.input_path if args.input_path else 0
video_name = os.path.basename(str(input_path))
vc = cv2.VideoCapture(input_path) #"..\\..\\Input_Videos\\GEYE0010_THM.mp4"
duration_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
print("Number of frames: " + str(duration_frames))

net = cv2.dnn.readNet(weight_file, cfg_file)
if not args.cpu:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(RESOLUTION, RESOLUTION), scale=1/255, swapRB=False)
model_name = "YOLOv4-"+ str(RESOLUTION)
print("Init done")

det_objects = []
widgets = [pb.Variable('FPS'), ', ', pb.Bar(), pb.Percentage()]
bar = pb.ProgressBar(max_value = duration_frames if duration_frames > 0 else pb.UnknownLength, redirect_stdout=True, widgets=widgets)

# Main loop over all video frames
# Stops if Ctrl+C is entered (KeyboardInterrupt)

frame_number = 0
while True:
    try:
        # Retrieve next video frame
        grabbed, frame = vc.read()
        if not grabbed:
            print("end")
            break
        height, width = frame.shape[:2]
        frame_number += 1

       # Perform detection on this frame
        start = time.time()
        classes, scores, bboxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
        end = time.time()

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(len(classes)):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in interesting_classes or bboxes[i][2] > 0.9 * width and bboxes[i][3] > 0.9 * height:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)

        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        if TRACK:
            features = encoder(frame, bboxes)
            detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

            #boxs = np.array([d.tlwh for d in detections])
            #scores = np.array([d.confidence for d in detections])
            #classes = np.array([d.class_name for d in detections])     

            # Call the tracker
            tracker.predict()
            tracker.update(detections)

            # update tracks
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue 
                box = track.to_tlwh()
                x_obj = float(box[0] + box[2] / 2)
                y_obj = float(box[1] + box[3])
                class_name = track.get_class()
                track_id = int(track.track_id)
                obj = {
                            "video_source": video_name,
                            "detection_model": model_name,
                            "frame_number": frame_number,
                            "class_title": class_name,
                            "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                            "coordinates_image": {"x": x_obj, "y": y_obj},
                            "track_id": track_id
                        }
                det_objects.append(obj)
        else:
            # Format detections
            for (class_name, score, box) in zip(names, scores, bboxes):
                x_obj = float(box[0] + box[2] / 2)
                y_obj = float(box[1] + box[3])
                obj = {
                        "video_source": video_name,
                        "detection_model": model_name,
                        "frame_number": frame_number,
                        "class_title": class_name,
                        "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                        "coordinates_image": {"x": x_obj, "y": y_obj}
                    }
                det_objects.append(obj)

        # FPS_value = 1 / (end - start)
    	# fps_label = "FPS: %.2f" % FPS_value
        # bar.update(frame_number, FPS=FPS_value)
    except KeyboardInterrupt:
        break

endname = "_output.json" if not args.track else "_track_output.json"
outname = os.path.basename(input_path).split('.')[0] + endname
store_result_detections = store_bson(det_objects, outname)


cv2.waitKey()
vc.release()
cv2.destroyAllWindows()
