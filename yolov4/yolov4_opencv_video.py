import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import time
import argparse
import numpy as np
import progressbar as pb

def parser():
    parser = argparse.ArgumentParser(description="Object Detection Algorithm Evaluation")
    parser.add_argument("-i", "--input_path", type=str, default='',
                        help="Path to the video on which detection must be performed. Default is 0 for the computer's webcam.")
    parser.add_argument("-cl", "--classes", type=str, default='',
                        help="Classes of interest. Default: all COCO classes. Eg. 'person;cell phone;car' (without blanks between classes).")
    parser.add_argument("-s", "--save_video", action='store_true',
                        help="Save video output to mp4 file.") 
    parser.add_argument("-o", "--out_path", type=str, default='output/yolov4-output/',
                        help="Path for the output video file. Default is 'output/yolov4-output/")
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
    parser.add_argument("-sh", "--show", action='store_true',
                        help="Show video output.") 
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
    
if not os.path.exists(args.out_path):
    os.makedirs(args.out_path)

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
class2color = {
    'person': (187,139,65),
    'car': (153, 51, 255),
    'bicycle': (80,150,117),
    'motorbike': (153, 51, 255),
    'truck': (153, 51, 255),
    'bus': (153, 51, 255),
}

interesting_classes = []
if args.classes:
    interesting_classes = args.classes.split(';')

class_names = []
with open(os.path.join("yolo-coco-data","coco.names"), "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

if not interesting_classes:
    interesting_classes = class_names

input_path = args.input_path if args.input_path else 0
vc = cv2.VideoCapture(input_path) #"..\\..\\Input_Videos\\GEYE0010_THM.mp4"
duration_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
print("Number of frames: " + str(duration_frames))


writer = None
weight_file = args.weights
cfg_file = args.config if args.config else os.path.join("yolo-coco-data","yolov4.cfg")

net = cv2.dnn.readNet(weight_file, cfg_file)
if not args.cpu:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(RESOLUTION, RESOLUTION), scale=1/255, swapRB=False)
model_name = "YOLOv4-"+ str(RESOLUTION)
print("Init done")

widgets = [pb.Variable('FPS'), ', ', pb.Bar(), pb.Percentage()]
bar = pb.ProgressBar(max_value = duration_frames if duration_frames > 0 else pb.UnknownLength, redirect_stdout=True, widgets=widgets)
f_num = 0
while cv2.waitKey(1) < 1:
    grabbed, frame = vc.read()
    if not grabbed:
        print("exit")
        break
    f_num += 1
    height, width = frame.shape[:2]

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
            class_name = track.get_class()
            track_id = int(track.track_id)
            #color = COLORS[int(classid) % len(COLORS)]
            color = class2color[class_name] if class_name in class2color else (255, 0, 0)
            #label = "%s : %f" % (class_names[classid[0]], score)
            label = "%s - %s" % (class_name, track_id)
            cv2.rectangle(frame, box, color, 2)
            cv2.putText(frame, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    else:
        # Format detections
        for (class_name, score, box) in zip(names, scores, bboxes):
            #color = COLORS[int(classid) % len(COLORS)]
            color = class2color[class_name] if class_name in class2color else (255, 0, 0)
            #label = "%s : %f" % (class_names[classid[0]], score)
            label = "%s" % (class_name)
            cv2.rectangle(frame, box, color, 2)
            cv2.putText(frame, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    FPS_value = 1 / (end - start)
    fps_label = "FPS: %.2f" % FPS_value
    bar.update(f_num, FPS=FPS_value)

    if args.show:
        cv2.namedWindow('YoloV4 Detection', cv2.WINDOW_NORMAL)
        cv2.imshow("YoloV4 Detection", frame)

    if args.save_video and writer is None:
        # Constructing code of the codec
        # to be used in the function VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # Writing current processed frame into the video file
        # Pay attention! If you're using Windows, yours path might looks like:
        # r'videos\result-traffic-cars.mp4'
        # or:
        # 'videos\\result-traffic-cars.mp4'
        outname = os.path.basename(input_path).split('.')[0] + "_output.mp4"
        writer = cv2.VideoWriter(args.out_path + outname, fourcc, 30,
                                    (frame.shape[1], frame.shape[0]), True)

    if args.save_video:
        # Write processed current frame to the file
        writer.write(frame)

cv2.waitKey()
vc.release()
cv2.destroyAllWindows()
