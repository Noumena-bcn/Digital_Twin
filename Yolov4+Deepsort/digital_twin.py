import os
import cv2
import argparse
import json
from copy import copy
import matplotlib.pyplot as plt
import numpy as np

# flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')

def parser():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-iv", "--input_video_path", type=str, default='',
                    help="Path to the video.")
    parser.add_argument("-id", "--input_detections_path", type=str, default='',
                    help="Path to the detections file. E.g. '..\\..\\Input_Videos\\GEYE0010_THM.mp4'. Default is 0 (webcam).")
    parser.add_argument("-o", "--output_path", type=str, default='.',
                    help="Path to the video output.")
    parser.add_argument("-cl", "--classes", type=str, default='',
                        help="Classes of interest. Default: all COCO classes. Eg. 'person;cell phone;car' (without blanks between classes).")
    parser.add_argument("-ds", "--dont_show", action='store_true',
                    help="Don't show video output.")
    parser.add_argument("-v", "--verbose", action='store_true',
                    help="Show detailed info of tracked objects.")
    # parser.add_argument("-s", "--save", action='store_true',
    #                     help="SSave the output as a mp4 video.")
    return parser.parse_args()



def dict2tuple(detdict):
        fn = detdict['frame_number']
        bbox0 = detdict['bbox']
        bbox = [bbox0[0], bbox0[1], bbox0[0]+bbox0[2], bbox0[1]+bbox0[3]]
        classT = detdict['class_title'] 
        try:
           track_id = detdict['track_id']
        except:
            track_id = 0

        return (classT, bbox, track_id, fn)

def import_detections(file_path, max_frame):
    filename = os.path.basename(file_path)

    with open(file_path) as fd:
        content = json.load(fd)
        fd.close()

    data = [[['class', 'bbox', 'track_id', 'frame_num']]]

    f = 1
    frame_list = []
    for det in content:
        if det["frame_number"] == f:
            frame_list.append(dict2tuple(det))
        else:
            while det["frame_number"] - f > 1:
                data.append([()])
                f+=1
            if det["frame_number"] - f == 1: 
                f = det["frame_number"]
                data.append(copy(frame_list))
                frame_list = [dict2tuple(det)]
            else:
                print('strange')
    data.append(copy(frame_list))

    while len(data) <= max_frame:
        print('padding')
        data.append([()])

    return data


def main():
    args = parser()
    print(args)

    video_path = args.input_video_path
    det_path = args.input_detections_path
    
    # video_path = r"C:\Users\oriol\Desktop\Noumena\Python\Digital_Twin\Videos entrada (5)\*.mp4"

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None

    duration_frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)
    detections = import_detections(det_path, duration_frames)

    # get video ready to save locally if flag is set
    if args.output_path:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*'mp4v')
        outname = os.path.basename(video_path).split('.')[0] + "_output.mp4"
        out = cv2.VideoWriter(os.path.join(args.output_path, outname), codec, fps, (width, height))

    frame_num = 0
    # while video is running
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            print('Video has ended or failed.')
            break
        frame_num +=1

        map = cv2.imread('map.png')
        print('Frame #: ', frame_num)
        frame_size = frame.shape[:2]
        print('Frame size ', frame_size)
        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        ######################
        

        resized = cv2.resize(frame, (1684, 945), interpolation = cv2.INTER_AREA)
            
        ################ CAM4 ################
        # pts1 = np.float32([[230,772], [384, 800], [117, 858], [279, 905]])
        # pts2 = np.float32([[237,724],[341,724],[237,834],[341,841]])
        ######################################

        pts1 = np.float32([[993,538], [1171,565], [754, 755], [1123, 829]])
        pts2 = np.float32([[1120+100,530], [1120+100,813], [220+100,530],[220+100,813]])//2
        matrix = cv2.getPerspectiveTransform(pts1,pts2)
        imgOutput = cv2.warpPerspective(resized,matrix,(1684, 945))

        h_img_out, w_img_out = imgOutput.shape[:2]
        imgOutput = imgOutput[0:0+int(h_img_out/2), 0:0+int(w_img_out-w_img_out/4)]

         # scale_percent = 20 # percent of original size   CAM4
        scale_percent = 53 # percent of original size   CAM9
        width = int(imgOutput.shape[1] * scale_percent / 100)
        height = int(imgOutput.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized_ = cv2.resize(imgOutput, dim, interpolation = cv2.INTER_AREA)

        #############################################################

        #CAM4
        # x = 120
        # y = 970

        #CAM9
        y = 365
        x = 285


        #############################################################
        print("prova", resized_.shape)
        print(map.shape)
        print(map[y:y+resized_.shape[0], x:x+resized_.shape[1]].shape)
        imgToblend = np.ones_like(map)
        imgToblend[y:y+resized_.shape[0], x:x+resized_.shape[1]] =  resized_
        
        cv2.cvtColor(imgToblend, cv2.COLOR_BGR2RGB)
        map = cv2.addWeighted(map, 1, imgToblend, 0.5, 0)

        # detections = [[['class', 'bbox', 'track_id', 'frame_num']]]
        for det in detections[frame_num]:
            if not det:
                continue
            class_name = det[0]
            bbox = det[1]
            track_id = det[2]
            
        # draw bbox on screen
            color = colors[int(track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)

            
            ############################## Extract detection #####################   Utilitzar els cercles per veure on assenyalen les coordenades

            facePoint1 = (bbox[0] + bbox[2]) / 2
            facePoint2 = (bbox[3] + bbox[1]) / 2 
            centerX = int(facePoint1)
            centerY = int(facePoint2)  
            
            
            height_det = int(bbox[2] - bbox[0]) 
            width_det = int(bbox[3] - bbox[1])   

            # print("height_det")
            # print(height_det)
            # print("width_det")
            # print(width_det)

        

            # final_face = frame[centerY-height_det:centerY+height_det,centerX-width_det:centerX+width_det]
            # cv2.imshow("Final face",final_face)

            #######################################################################

            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track_id)))*17, int(bbox[1])), color, -1)
            cv2.putText(frame, class_name + "-" + str(track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
            cv2.circle(frame, (centerX, int(bbox[3])), 10, (255, 0, 0), 2)

            # resized = cv2.resize(frame, (1684, 945), interpolation = cv2.INTER_AREA)
            
            # ################ CAM4 ################
            # # pts1 = np.float32([[230,772], [384, 800], [117, 858], [279, 905]])
            # # pts2 = np.float32([[237,724],[341,724],[237,834],[341,841]])
            # ######################################

            # pts1 = np.float32([[993,538], [1171,565], [754, 755], [1123, 829]])
            # pts2 = np.float32([[1120+100,530], [1120+100,813], [220+100,530],[220+100,813]])//2
            # matrix = cv2.getPerspectiveTransform(pts1,pts2)
            # imgOutput = cv2.warpPerspective(resized,matrix,(1684, 945))
            
            # h_img_out, w_img_out = imgOutput.shape[:2]
            # imgOutput = imgOutput[0:0+int(h_img_out/2), 0:0+int(w_img_out-w_img_out/4)]
            # # cv2.imshow('Test', imgOutput)

            # Resize points cause of different shapes
            lowX = int(centerX*1684/1280)
            lowY = int(int(bbox[3])*945/720)
            points = []
            points.append([lowX, lowY])
            # print("Points in", points)

            points = np.asarray(points, dtype=np.float32)
            points = np.array([points])

            pointsOut = cv2.perspectiveTransform(points, matrix)
            
            
            pointsOut = np.asarray(pointsOut)
            
            pointsOut = pointsOut.tolist()[0]

            pointsOut =  pointsOut[0]
            pointsOut0 =  pointsOut[0]
            pointsOut1 =  pointsOut[1]

            centerX_conv = abs(int(pointsOut0))
            centerY_conv = abs(int(pointsOut1))


            cv2.circle(imgOutput, (centerX_conv, centerY_conv), 10, (255, 0, 255), 2)

            # # scale_percent = 20 # percent of original size   CAM4
            # scale_percent = 53 # percent of original size   CAM9
            # width = int(imgOutput.shape[1] * scale_percent / 100)
            # height = int(imgOutput.shape[0] * scale_percent / 100)
            # dim = (width, height)
            # resized_ = cv2.resize(imgOutput, dim, interpolation = cv2.INTER_AREA)

            # #############################################################

            # #CAM4
            # # x = 120
            # # y = 970

            # #CAM9
            # x = 345
            # y = 220


            # #############################################################
            # map[x:x+resized_.shape[0], y:y+resized_.shape[1]] =  resized_
            # print("prova", resized_.shape)
            #CAM4
            # Ymap = (189*centerX_conv/945) 
            # Xmap = (336*centerY_conv/1684) 

            #CAM9
            # Ymap = (250*centerX_conv/945) 
            # Xmap = (669*centerY_conv/1684)
            Xmap = (resized_.shape[1]*centerX_conv/imgOutput.shape[1]) 
            Ymap = (resized_.shape[0]*centerY_conv/imgOutput.shape[0])

            mapY = int(y + Ymap)
            mapX = int(x + Xmap)
            # print("mapY", mapX)
            # print("mapX", mapY)
            print("centerX_conv", centerX_conv)
            print("centerY_conv", centerY_conv)

            cv2.circle(map, (mapX, mapY), 5, (255, 0, 255), -1)

            # if enable info flag then print details about each track
            if args.verbose:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

        print("Image shape frame", map.shape)
        # print("Image shape imgout", imgOutput.shape)
        cv2.imshow("Resized", imgOutput)
        # cv2.imshow("Floor", map)
            
            

        #############################################################
        # filename = "frame_floor_cam9\{}.jpg".format(frame_num)
        # print("FINAL FACE",filename)
        
        # cv2.imwrite(filename, map)
            
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    

        cv2.imshow("Floor", map)
        # if not FLAGS.dont_show:
        #     cv2.imshow("Output Video", result)
        cv2.imshow("Output Video", result)
        # if output flag is set, save video file
        if args.output_path:
            out.write(map)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass