"""
Use this module to perform operations on a MongoDB database using the pymongo driver.

Functions:
-store_bson
-store_bson2
-import_json2mongo
-extract_path_info
-store_mongo
-get_video_meta
-update_video_meta
-store_markers_frame
-store_markers_geo
-copy_markers
-update_to_geojson
-update_detections_geo
-get_center_geo
-get_detections_geo
-find_rider
-make_index
-drop_collection
-delete_docs
-count_docs

Required:
-numpy
-pymongo
-json
-datetime
-os
-argparse
-time

-my_client_name.py (MONGO_CLIENT_NAME variable = client name stored in an additionnal file so it's not exposed on GitHub)
-transformations.py

"""

import numpy as np
from pymongo import MongoClient, UpdateOne, GEO2D, GEOSPHERE
import json
import datetime as dt
import os
import argparse
import time

from my_client_name import MONGO_CLIENT_NAME
import transformations as tools

DEFAULT_DB = 'TestDT'
DEFAULT_CLIENT = MongoClient(MONGO_CLIENT_NAME)
DEFAULT_CRS = "EPSG3857"

def parser():
    parser = argparse.ArgumentParser(description="Custom MongoDB Driver for TestDT based on pymongo")
    parser.add_argument("-i", "--input_path", type=str, default='',
                        help="Path to the video on which detection must be performed. E.g. '..\\..\\Input_Videos\\GEYE0010_THM.mp4'. Default is 0 (webcam).")
    return parser.parse_args()

def store_bson(objects, file_name):
    """
    Stores dict objects in a JSON file.
    Objects can be either one dict object or a list of dict objects.
    """
    with open(file_name, 'w') as f:
        json.dump(objects, f)
        f.close()

def store_bson2(objects, file_name):
    """
    Stores dict objects in a JSON file.
    Objects can be either one dict object or a list of dict objects.
    """
    store = ""
    for obj in objects:
        store += json.dumps(obj)
        store += '\n'

    with open(file_name, 'w') as f:
        f.write(store)
        f.close()

def import_json2mongo(file_name, collection, database=DEFAULT_DB, client=DEFAULT_CLIENT):
    """
    Loads a JSON file and stores it as one or several documents in a MongoDB database.
    The JSON file must respect: one JSON object <-> one line
    """
    client = client if client else MongoClient(MONGO_CLIENT_NAME)
    with open(file_name) as f:
        data = json.load(f)
        f.close()
    store_mongo(data, collection, database, client)

def extract_path_info(path, root_included='/media/noumena/noumena/URBAN MAPPING/'):
    '''
    input: path string (eg: "/media/noumena/noumena/URBAN MAPPING/Camara 1 - Consell rambla/2020-10-04/mp4_g/17.46.01-18.32.08[R][0@0][0].mp4")
    output: dict
        title: string (eg: 'ConsellRambla_2020-10-04_17-46')
        location_name: string (eg: 'Camara1Consellrambla')
        start_date: datetime.datetime
        end_date: datetime.datetime
    '''
    output = {}
    if root_included in path:
        print("Reading a video from '{}', the metadata will be extracted from the provided path".format(root_included))

        path_list = path.split('/')[5:]
        loc_title = path_list[0]
        loc_title = ''.join(''.join(''.join(loc_title.split(' ')).split('(')).split(')')[0].split('-'))
        date = path_list[1].split('-')
        start_time, end_time = path_list[3].split('[',1)[0].split('-')

        start_time = start_time.split('.')
        end_time = end_time.split('.')

        output['title'] = '_'.join([loc_title, path_list[1], '-'.join(start_time[:2])])
        output['location_name'] = loc_title
        output['start_date'] = dt.datetime(*[int(d) for d in date], *[int(t) for t in start_time])
        output['end_date'] = dt.datetime(*[int(d) for d in date], *[int(t) for t in end_time])

        return output
    else:
        print('Reading a video from an unknown path, only the basename will be returned.')
        output['title'] = os.path.basename(path).split('.')[0]
        output['location_name'] = None
        output['start_date'] = None
        output['end_date'] = None

        return output

def store_mongo(objects, collection, database=DEFAULT_DB, client=DEFAULT_CLIENT):
    """
    Input:
    objects: list of dict objects compatible with JSON format.
    
    InsertMany request to client.database.collection if objects is not empty.
    """
    client = client if client else MongoClient(MONGO_CLIENT_NAME)
    if objects:
        print("\nStored {} objects in {}.{}\n".format(len(objects), database, collection))
        return client[database][collection].insert_many(objects)
    else:
        print("Nothing to store in MongoDB.")
        return None

def get_video_meta(video_name, database=DEFAULT_DB, client=DEFAULT_CLIENT):
    """
    Retrieve video metadata from the database.
    Return a dict holding the metadata. 
    """
    client = client if client else MongoClient(MONGO_CLIENT_NAME)
    return client[database]["videosMeta"].find_one({"video_source": video_name})

def update_video_meta(video_name, video_meta, database=DEFAULT_DB, client=DEFAULT_CLIENT):
    """
    video_name: str (e.g. "GEYE0005.MP4").
    video_meta: dict object compatible with JSON format.
    video_meta keys can be:
        -video_source (document created if doesn't exist)
        -frame_height
        -frame_width
        -duration_frames
        -location_name
        -camera_id (if there are several cameras at the same location)
    UpdateOne request to client.urbanBcnMappingDB.videosMeta with {"video_source": video_name} as filter.
    """
    client = client if client else MongoClient(MONGO_CLIENT_NAME)
    return client[database]["videosMeta"].update_one({"video_source": video_name}, {"$set": video_meta}, upsert=True)

def store_markers_frame(ordered_markers, video_name, database=DEFAULT_DB, client=DEFAULT_CLIENT):
    """
    Takes a (4, 2) ndarray or list of markers (x,y) coordinates ordered as:
    1: top-left,
	2: top-right,
	3: bottom-right,
	4: bottom-left
    Stores these coordinates as [x,y] pairs in the MongoDB document that has video_name as value for the video_source key.
    The document is created if it doesn't exist.
    Returns the output value of the pymongo update_one() method.
    """
    client = client if client else MongoClient(MONGO_CLIENT_NAME)
    db = client[database]
    col = db["videosMeta"]
    markers = {
        "tl": [float(ordered_markers[0][0]), float(ordered_markers[0][1])],
        "tr": [float(ordered_markers[1][0]), float(ordered_markers[1][1])],
        "br": [float(ordered_markers[2][0]), float(ordered_markers[2][1])],
        "bl": [float(ordered_markers[3][0]), float(ordered_markers[3][1])]
        }
    return col.update_one({"video_source": video_name}, {"$set": {"markers_frame": markers}}, upsert=True)

def store_markers_geo(ordered_markers, video_name, geo_system_name=DEFAULT_CRS, database=DEFAULT_DB, client=DEFAULT_CLIENT):
    """
    Takes a (4, 2) ndarray or list of markers (lat,lng) coordinates ordered as:
    1: top-left,
	2: top-right,
	3: bottom-right,
	4: bottom-left
    Stores these coordinates as GeoJSON geometry objects in the MongoDB document that has video_name as value for the video_source key.
    The document is created if it doesn't exist.
    Returns the output value of the pymongo update_one() method.
    """
    client = client if client else MongoClient(MONGO_CLIENT_NAME)
    db = client[database]
    col = db["videosMeta"]
    markers = {
        geo_system_name:{
            "tl": {"type": "Point", "coordinates": [float(ordered_markers[0][1]), float(ordered_markers[0][0])]},
            "tr": {"type": "Point", "coordinates": [float(ordered_markers[1][1]), float(ordered_markers[1][0])]},
            "br": {"type": "Point", "coordinates": [float(ordered_markers[2][1]), float(ordered_markers[2][0])]},
            "bl": {"type": "Point", "coordinates": [float(ordered_markers[3][1]), float(ordered_markers[3][0])]}
        }
    }
    return col.update_one({"video_source": video_name}, {"$set": {"markers_geo": markers}}, upsert=True)

def copy_markers(video_name_source, video_name_dest, database=DEFAULT_DB, client=DEFAULT_CLIENT):
    """
    Copy the markers (frame markers and geo markers) if they exist from a video to another video in the videometa collection.
    """
    client = client if client else MongoClient(MONGO_CLIENT_NAME)
    db = client[database]
    col = db["videosMeta"]

    video_meta_source = col.find_one({"video_source": video_name_source})
    try:
        markers_frame = video_meta_source["markers_frame"]
    except Exception as e:
        print(e)
        print("No 'markers_frame' found for " + video_name_source)
    try:
        markers_geo = video_meta_source["markers_geo"]
    except Exception as e:
        print(e)
        print("No 'markers_geo' found for " + video_name_source)
    print("copying markers")
    return col.update_one({"video_source": video_name_dest}, {"$set": {"markers_frame": markers_frame, "markers_geo": markers_geo}}, upsert=True)

def update_to_geojson(video_name, geo_system_name=DEFAULT_CRS, database=DEFAULT_DB, client=DEFAULT_CLIENT):
    """
    If the detections for a video are stored using the previous Noumena format, change this format to GeoJSON.
    """
    client = client if client else MongoClient(MONGO_CLIENT_NAME)
    db = client[database]
    colDet = db[video_name.split('.', 1)[0]]

    def _update_im(detections_cursor):
        try:
            dets = [(det["_id"], [det["coordinates_image"]["x"], det["coordinates_image"]["y"]]) for det in detections_cursor]
        except Exception as e:
            print("Error in fetching coordinates: ", e)
            return []
        det_ids = [d[0] for d in dets]
        det_pts = [d[1] for d in dets]
        requests = []
        for i in range(len(det_ids)):
            requests.append(UpdateOne({"_id": det_ids[i]}, {"$set": {"coordinates_image": det_pts[i]}})) #[x,y]
        return requests

    def _update_geo(detections_cursor):
        try:
            dets = [(det["_id"], [det["coordinates_geo."+geo_system_name]["lng"], det["coordinates_geo."+geo_system_name]["lat"]]) for det in detections_cursor]
        except Exception as e:
            print("Error in fetching coordinates: ", e)
            return []
        det_ids = [d[0] for d in dets]
        det_pts = [d[1] for d in dets]
        requests = []
        for i in range(len(det_ids)):
            requests.append(UpdateOne({"_id": det_ids[i]}, {"$set": {"coordinates_geo."+geo_system_name: {"type":"Point", "coordinates":det_pts[i]}}})) #[lng, lat]
        return requests

    nb_det = colDet.count_documents({})
    withoutGeo = {"coordinates_geo."+geo_system_name: {"$exists": False}}
    withGeo = {"coordinates_geo."+geo_system_name: {"$exists": True}}
    nb_det_geo = colDet.count_documents(withGeo)

    requests = []
    if not nb_det:
        print("Nothing to update")
        return None
    elif nb_det == nb_det_geo:
        detections_cursor_tot = colDet.find({})
        requests = _update_im(detections_cursor_tot)
        requests += _update_geo(detections_cursor_tot)
    else:
        det_cursor = colDet.find(withoutGeo)
        requests = _update_im(det_cursor)
        if nb_det_geo:
            det_cursor_geo = colDet.find(withGeo)
            requests += _update_geo(det_cursor_geo)

    print("Updating the database...")
    if requests:
        write_result = colDet.bulk_write(requests).bulk_api_result
        print("Update done")
    else:
        print("No update request to send")
        write_result = None
    return write_result

def update_detections_geo(video_name, geo_system_name=DEFAULT_CRS, force_update=False, database=DEFAULT_DB, client=DEFAULT_CLIENT):
    """
    This function maps the detection coordinates from the video frame to the geographic location.
        - Read frame markers and geo markers (of the specified CRS) from videosMeta MongoDB collection
        - Find the homography matrix between these markers
        - Fetch all the detections from the MongoDB collection of the specified video that aren't already mapped
        - Update these detections with geographic coordinates.
        - Create a 2dsphere index on the GeoJSON coordinates.
    video_name: str (e.g. "GEYE0005.MP4")
    geo_system_name: str (default "EPSG3857")
    force_update: bool (default False), used to overwrite any preexisting coordinates.
    """
    client = client if client else MongoClient(MONGO_CLIENT_NAME)
    db = client[database]
    colDet = db[video_name.split('.', 1)[0]]
    colVid = db["videosMeta"]

    video_meta = colVid.find_one({"video_source": video_name, "markers_geo."+geo_system_name: {"$exists": True}})
    if video_meta:
        print("Video meta:")
        print(video_meta)
        print()
        corners_img = [[video_meta["markers_frame"][pt][0], video_meta["markers_frame"][pt][1]] for pt in video_meta["markers_frame"]]
        corners_geo = [[video_meta["markers_geo"][geo_system_name][pt]["coordinates"][0], video_meta["markers_geo"][geo_system_name][pt]["coordinates"][1]] for pt in video_meta["markers_geo"][geo_system_name]]

        H = tools.four_point_transform_geo(corners_img, corners_geo)

        filter = {"coordinates_geo."+geo_system_name: {"$exists": False}}
        nb_det = colDet.count_documents(filter)
        print("Found {} detection documents without geographic coordinates for CRS:{}".format(nb_det, geo_system_name))
        if force_update: print("'force_update=True, updating detections anyway")
        if force_update or nb_det:
            print("Fetching the database...")
            detections_cursor = colDet.find() if force_update else colDet.find(filter)
            dets = [(det["_id"], [det["coordinates_image"][0], det["coordinates_image"][1]]) for det in detections_cursor]
            det_ids = [d[0] for d in dets]
            det_pts = [d[1] for d in dets]
            #print(det_ids)
            #print(det_pts)
            print("Performing homography...")
            pts_geo = tools.homography(det_pts, H)
            #print(pts_geo)
            requests = []
            for i in range(len(det_ids)):
                requests.append(UpdateOne({"_id": det_ids[i]}, {"$set": {"coordinates_geo."+geo_system_name: {"type":"Point", "coordinates": [pts_geo[i][0], pts_geo[i][1]]}}}))
            print("Updating the database...")
            write_result = colDet.bulk_write(requests)
            print("Update done")
            make_index(video_name, type="geo")
            return write_result
        else:
            print("Nothing to update")
            return None
    else:
        print("No video found in the database with following infos:")
        print("video_source: "+video_name)
        print("CRS: "+geo_system_name)
        return None

def get_center_geo(video_name, geo_system_name=DEFAULT_CRS, database=DEFAULT_DB, client=DEFAULT_CLIENT):
    """
    Return the geographic center [lng, lat] of the geo markers defined for the video.
    video_name: str (e.g. "GEYE0005.MP4")
    geo_system_name: str (default "EPSG3857")
    """
    client = client if client else MongoClient(MONGO_CLIENT_NAME)
    db = client[database]
    colMeta = db["videosMeta"]
    videoMeta = colMeta.find_one({"video_source": video_name})

    markers = [ [videoMeta["markers_geo"][geo_system_name][m]["coordinates"][0], videoMeta["markers_geo"][geo_system_name][m]["coordinates"][1]] for m in videoMeta["markers_geo"][geo_system_name]]

    center = [np.mean(markers, axis=0)[0], np.mean(markers, axis=0)[1]]
    return center

def get_detections_geo(video_name, geo_system_name=DEFAULT_CRS, classTitle=None, filter=None, ignore_geo=False, database=DEFAULT_DB, client=DEFAULT_CLIENT):
    """
    Fetch all detections whith geo coordinates matching the given filters in the MongoDB collection of the specified video
    video_name: str (e.g. "GEYE0005.MP4")
    geo_system_name: str (default "EPSG3857")
    classTitle: str (e.g. "person")
    filter: dict (e.g. {"frame_number": {"$gt": 30, "$lte": 80}}
    ignore_geo: bool (default False), fetch also detections without geo coordinates.
    """
    client = client if client else MongoClient(MONGO_CLIENT_NAME)
    db = client[database]
    colDet = db[video_name.split('.', 1)[0]]

    filter = filter if filter else {}
    if classTitle:
        filter["class_title"]=classTitle
    if "coordinates_geo" not in filter and not ignore_geo:
        filter["coordinates_geo."+geo_system_name] = {"$exists": True}

    det_cursor = colDet.find(filter).sort("frame_number", 1)

    docs = [d for d in det_cursor]
    if not docs:
        print("No detection found matching this filter:")
        print(filter)
    return docs

def find_rider(video_name, vehicle_class, udpate_db=False, database=DEFAULT_DB, client=DEFAULT_CLIENT):
    """
    For each detection of the 'vehicle_class', find the closest 'person' detection in the image.
    By default only stats about the detections are displayed.
    To update the 'person' detection identified as the rider, set update_db=True.
    """
    client = client if client else MongoClient(MONGO_CLIENT_NAME)
    db = client[database]
    colDet = db[video_name.split('.', 1)[0]]
    print("Video name: ", video_name)

    if not 'coordinates_image_2d' in [index["name"] for index in colDet.list_indexes()]:
        print("This collection doesn't have a 2d index on 'coordinates_image'")
        return None

    print("Fetching {}...".format(vehicle_class))
    veh_cursor = get_detections_geo(video_name, classTitle=vehicle_class, ignore_geo=True, filter={"frame_number":{"$lte": 1000}})
    veh_objs = [(det["frame_number"], [det["coordinates_image"][0], det["coordinates_image"][1]], det["detection_height"], det["_id"]) for det in veh_cursor]

    print("Calculating distances...")
    dists = []
    requests = []
    for vehi in veh_objs:
        #filter={"class_title": "person", "frame_number": vehi[0], "coordinates_image": {"$geoWithin": {"$center": [vehi[1], 500]}}}
        # start = time.time()
        filter={"class_title": "person", "frame_number": vehi[0], "coordinates_image": {"$near": vehi[1]}}
        rider = colDet.find_one(filter)
        if rider:
            dist = np.sqrt((vehi[1][0]-rider["coordinates_image"][0])**2+(vehi[1][1]-rider["coordinates_image"][1])**2)
            if dist < 0.7* vehi[2]:
                dists.append(dist)
                requests.append(UpdateOne({"_id": rider["_id"]}, {"$set": {"riding": {"class_title": vehicle_class, "_id": vehi[3]}}}))
        # end = time.time()
        # print(end-start)
    print("Calculating stats...")
    print(len(dists))
    if dists:
        print(np.min(dists))
        print(np.max(dists))
        print(np.mean(dists))
        print(np.std(dists))
    if udpate_db:
        print("Updating the database...")
        result = colDet.bulk_write(requests).bulk_api_result
        return result
    return len(dists)

def make_index(video_name, type, geo_system_name=DEFAULT_CRS, database=DEFAULT_DB, client=DEFAULT_CLIENT):
    """
    Create a 2d (coordinates_image) or 2dsphere (coordinates_geo) index for the given collection based on the given type
    type: str, "image"=> 2d or "geo"=>2dsphere 
    """
    client = client if client else MongoClient(MONGO_CLIENT_NAME)
    db = client[database]
    colDet = db[video_name.split('.', 1)[0]]
    print("Video name: ", video_name)

    if type=="image":
        video_meta = get_video_meta(video_name, database=database)
        max_dim = max(video_meta["frame_width"], video_meta["frame_height"])
        print(colDet.create_index([("coordinates_image", GEO2D)], min=-max_dim, max=1.5*max_dim))
        print("index created")
    elif type=="geo":
        print(colDet.create_index([("coordinates_geo."+geo_system_name, GEOSPHERE)]))
        print("index created")
    else:
        print("Argument type must be either 'image' or 'geo'")
        print("Argument type given: ", type)
    return None

def drop_collection(col, database=DEFAULT_DB, client=DEFAULT_CLIENT):
    """
    Drops the given collection.
    Returns True if success, False if collection doesn't exist.
    """
    client = client if client else MongoClient(MONGO_CLIENT_NAME)
    return client[database].drop_collection(col)

def delete_docs(filter, col, database=DEFAULT_DB, client=DEFAULT_CLIENT):
    """
    Delete documents matching the given filter in the given collection.
    """
    client = client if client else MongoClient(MONGO_CLIENT_NAME)
    return client[database][col].delete_many(filter)

def count_docs(filter, col, database=DEFAULT_DB, client=DEFAULT_CLIENT):
    """
    Counts documents matching the given filter in the given collection.
    """
    client = client if client else MongoClient(MONGO_CLIENT_NAME)
    return client[database][col].count_documents(filter)

if __name__ == "__main__":
    print("Coucou")
    video_n = "Camara1Consellrambla_2020-10-04_17-46"
    video_n = "Camara1Consellrambla_2020-10-04_20-45"
    #video_n = "Càmara3RocafortambGranVia_2020-10-11_17-00"
    #video_n = "Camara1Consellrambla_2020-10-04_19-18"
    video_s = "bcn_afternoon.mp4"

    args=parser()
    video_name = extract_path_info(args.input_path)['title']

    # find_rider(video_name, "motorbike")

    # make_index(video_name, 'geo')
    # make_index(video_name, 'image')

    # copy_markers(video_s, video_name)
    # print(update_to_geojson(video_name))

    update_detections_geo(video_name, "EPSG3857")
    #dets = get_detections_geo(video_n, classTitle="car", filter ={"frame_number": {"$lt": 22, "$gt": 5}})
    #print(dets)

    # client = MongoClient(MONGO_CLIENT_NAME)
    # db = client["urbanBcnMappingDB"]
    # col = db[video_name]



    # col = db["Times_Square_2021-04-25_22-05-30"]
    # print(col.update_many({}, {"$unset": {"driving":""}}))
