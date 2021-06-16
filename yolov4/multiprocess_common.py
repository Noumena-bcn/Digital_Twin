import multiprocessing
import digital_twin_func as dt
import map_prova as mp
from multiprocessing import Pipe
import cv2
import argparse
import numpy as np
import mongo as stm
import math

# def parser():
#     parser = argparse.ArgumentParser(description="")
#     parser.add_argument("-s", "--save", action='store_true',
#                         help="SSave the output as a mp4 video.")
#     return parser.parse_args()





if __name__ == "__main__":
    # args = parser()
    parent_conn, child_conn = Pipe()
    # parent_conn1, child_conn1 = Pipe()
    parent_conn2, child_conn2 = Pipe()
    parent_conn3, child_conn3 = Pipe()
    parent_conn4, child_conn4 = Pipe()
    parent_conn5, child_conn5 = Pipe()

    #  child_conn,
    cam2 = multiprocessing.Process(target= dt.main, args = ('CAM2','/home/noumena/Documents/DT_vid/vid_new/cam2_1_.mp4', '/home/noumena/Documents/Digital_Twin/Yolov4+Deepsort/yolo_json_files/cam2_1__output.json',child_conn,))
    # cam5 = multiprocessing.Process(target= dt.main, args = ('CAM5','/home/noumena/Documents/DT_vid/cam5.mp4', '/home/noumena/Documents/Digital_Twin/Yolov4+Deepsort/yolo_json_files/cam5_track_output.json',child_conn1,))
    cam6 = multiprocessing.Process(target= dt.main, args = ('CAM6','/home/noumena/Documents/DT_vid/vid_new/cam6_1_.mp4', '/home/noumena/Documents/Digital_Twin/Yolov4+Deepsort/yolo_json_files/cam6_1__output.json',child_conn2,))
    cam7 = multiprocessing.Process(target= dt.main, args = ('CAM7','/home/noumena/Documents/DT_vid/vid_new/cam7_1.mp4', '/home/noumena/Documents/Digital_Twin/Yolov4+Deepsort/yolo_json_files/cam7_1_output.json',child_conn3,))
    cam8 = multiprocessing.Process(target= dt.main, args = ('CAM8','/home/noumena/Documents/DT_vid/vid_new/cam8_1.mp4', '/home/noumena/Documents/Digital_Twin/Yolov4+Deepsort/yolo_json_files/cam8_1_output.json',child_conn4,))
    cam9 = multiprocessing.Process(target= dt.main, args = ('CAM9','/home/noumena/Documents/DT_vid/cam9.mp4', '/home/noumena/Documents/Digital_Twin/Yolov4+Deepsort/yolo_json_files/cam9_track_output.json',child_conn5,))
    

    cam2.start()

    # cam5.start()
    cam6.start()
    cam7.start()
    cam8.start()
    cam9.start()
    
    result = cv2.VideoWriter('map_output.mp4', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         15, (1422, 799))


    det_objects = []
    frame_num = 0
    col_name = "test6"

    while True:
        map_shape = cv2.imread('map.png')

        cam2_points = parent_conn.recv()
        cam6_points = parent_conn2.recv()
        cam7_points = parent_conn3.recv()
        cam8_points = parent_conn4.recv()
        cam9_points = parent_conn5.recv()
        
        for points2 in cam2_points:
            for points8 in cam8_points:
                zone = 1
                p2 = (points2[0], points2[1])
                p8 = (points8[0], points8[1])
                p2 = np.array(p6)
                p8 = np.array(p7)

                dist = int(np.linalg.norm(p8-p2))     

                if dist <= 15:
                    avg = np.mean( np.array([ p2, p8 ]), axis=0 )
                    avg = tuple(avg.astype(int))     

                    cv2.circle(map_shape, avg, 5, (43, 210, 88), -1)

                    cam2_points.remove(points2)
                    cam8_points.remove(points8)
                    obj = {
                                "Zone": float(zone),
                                "class_title": "person",
                               "detection_width_2": float(points6[2]),
                                "cam":{
                                    "cam2":{
                                        "coordinates":[float(points2[2]), float(points2[3])], "width": float(points2[4]), "height":float(points2[5])},
                                    "cam8":{
                                        "coordinates":[float(points8[2]), float(points8[3])], "width": float(points8[4]), "height":float(points8[5])}
                                        },
                                "coordinates_map": [float(avg[0]), float(avg[1])],
                                "frame_number": float(frame_num),
                            }
                    det_objects.append(obj)


        for points in cam2_points:
            points_floor = (points[0], points[1])
            cv2.circle(map_shape, points_floor, 5, (210, 43, 71), -1)
            zone = 1
            obj = {
                        "Zone": float(zone),
                        "class_title": "person",
                        "cam":{
                            "cam2":{
                                    "coordinates":[float(points[2]), float(points[3])], "width": float(points[4]), "height":float(points[5])
                                    }},
                        "coordinates_map": [float(points[0]), float(points[1])],
                        "frame_number": float(frame_num),
                    }
            det_objects.append(obj)

        for points in cam8_points:
            points_floor = (points[0], points[1])
            cv2.circle(map_shape, points_floor, 5, (216, 194, 52), -1)
            zone = 1
            obj = {
                        "Zone": float(zone),
                        "class_title": "person",
                        "cam":{
                            "cam8":{
                                    "coordinates":[float(points[2]), float(points[3])], "width": float(points[4]), "height":float(points[5])
                                    }},
                        "coordinates_map": [float(points[0]), float(points[1])],
                        "frame_number": float(frame_num),
                    }
            det_objects.append(obj)

        
        equal_p6 = []
        equal_p7 = []

        for points6 in cam6_points:
            for points7 in cam7_points:

                p6 =(points6[0], points6[1])
                p7 =(points7[0], points7[1])
                p6 = np.array(p6)
                p7 = np.array(p7)

                dist = int(np.linalg.norm(p6-p7)) 

                zone = 1

                if dist <= 15:
                    avg = np.mean( np.array([ p6, p7 ]), axis=0 )
                    avg = tuple(avg.astype(int))     

                    cv2.circle(map_shape, avg, 5, (255, 0, 255), -1)

                    
                    obj = {
                                "Zone": float(zone),
                                "class_title": "person",
                                "cam":{
                                    "cam6":{
                                        "coordinates":[float(points6[2]), float(points6[3])], "width": float(points6[4]), "height":float(points6[5])},
                                    "cam7":{
                                        "coordinates":[float(points7[2]), float(points7[3])], "width": float(points7[4]), "height":float(points7[5])}
                                        },
                                "coordinates_map": [float(avg[0]), float(avg[1])],
                                "frame_number": float(frame_num),
                            }
                    cam6_points.remove(points6)
                    cam7_points.remove(points7)
                    det_objects.append(obj)
                    

        

        for points in cam6_points:
            points_floor = (points[0], points[1])
            cv2.circle(map_shape, points_floor, 5, (52, 5, 25), -1)
            zone = 1
            obj = {
                        "Zone": float(zone),
                        "class_title": "person",
                        "cam":{
                            "cam6":{
                                    "coordinates":[float(points[2]), float(points[3])], "width": float(points[4]), "height":float(points[5])
                                    }},
                        "coordinates_map": [float(points[0]), float(points[1])],
                        "frame_number": float(frame_num),
                    }
            det_objects.append(obj)
        
        for points in cam7_points:
            points_floor = (points[0], points[1])
            cv2.circle(map_shape, points_floor, 5, (300, 300, 300), -1)
            zone = 1
            obj = {
                        "Zone": float(zone),
                        "class_title": "person",
                        "cam":{
                            "cam7":{
                                    "coordinates":[float(points[2]), float(points[3])], "width": float(points[4]), "height":float(points[5])
                                    }},
                        "coordinates_map": [float(points[0]), float(points[1])],
                        "frame_number": float(frame_num),
                    }
            det_objects.append(obj)

        for points in cam9_points:
            points_floor = (points[0], points[1])
            cv2.circle(map_shape, points_floor, 5, (255, 0, 255), -1)
            zone = 2

            obj = {
                        "Zone": float(zone),
                        "class_title": "person",
                        "cam":{
                            "cam9":{
                                    "coordinates":[float(points[2]), float(points[3])], "width": float(points[4]), "height":float(points[5])
                                    }},
                        "coordinates_map": [float(points[0]), float(points[1])],
                        "frame_number": float(frame_num),
                    }
            det_objects.append(obj)

        cv2.imshow("Final Result", map_shape)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        result.write(map_shape)

        frame_num += 1 
        # print("det_objects",det_objects)
        print("frame_num",frame_num)
        # if args.save:
        #     if writer is None:
        #         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        #         outname = 'map_ouput.mp4'
        #         writer = cv2.VideoWriter(outname, fourcc, 15, (map_shape.shape[1], map_shape.shape[0]), True)
        #     writer.write(map_shape)


        store_result_detections = stm.store_mongo(det_objects, col_name, database="TestDT")
        det_objects = []
    
    

    
    cam2.join()
    # cam5.join()
    cam6.join()
    cam7.join()
    cam8.join()
    cam9.join()
    
    
    cam9.join()

    print("Done!")