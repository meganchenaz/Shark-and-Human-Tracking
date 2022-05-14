import cv2
import numpy as np
import csv
import os
import math
from object_detection import ObjectDetection
from kalman_filter import KalmanFilter

def object_tracker(video_path, show_video, save_info, save_video, measure_success):
    """ NOTE: Images for measure_success will be overwritten (Be sure to move the images to 
        a new directory when running the program on multiple videos).
    """
    # initialize object detection
    od = ObjectDetection()

    # read in input video
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_fps = cap.get(cv2.CAP_PROP_FPS)

    # store the number of frames
    count = 0

    # store the previous frame's information
    info_prev_frame = []

    # store objects being tracked
    tracking_objects = {}
    track_id = 0

    # create distance threshold for tracking
    dist_threshold = min(vid_width, vid_height) // 10

    # store the class names
    class_names = ["Shark", "Human", "Human"]

    # if desired, generate output video
    if save_video == True:
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') 
        output_video = cv2.VideoWriter('output.avi', fourcc, vid_fps, (vid_width, vid_height))
    
    # if desired, create an info file
    if save_info == True:
        info_file = open("tracking_information.csv", "w", encoding='UTF8', newline='')
        writer = csv.writer(info_file)

        # write the header
        header = ['frame number', 'object id', 'object name', 'center x', 'center y', 'box width', 'box height']
        writer.writerow(header)
    
    if measure_success == True:
        # create a folder for saving the frames measuring success
        dir = 'images_for_measuring_success'
        if not os.path.exists(dir):
            os.mkdir(dir)
        file_index = 0

    while True:
        # read in one videoframe
        ret, frame = cap.read()

        # update the current number of frames 
        count += 1

        if not ret:
            print("The video does not exist or is done processing.")
            break

        # stores current frame's information
        info_cur_frame = []

        # detect objects in current frame
        (class_ids, scores, boxes) = od.detect(frame)

        index = 0
        for box in boxes:
            # compute the center point of each detected object
            (x, y, w, h) = box
            cx = int((x + x + w) / 2)
            cy = int((y + y + h) / 2)

            # create Kalman filter object
            kf = KalmanFilter(0.1, 1, 1, 1, 0.1,0.1)

            # add center point, bounding box information, class name, and Kalman filter object
            # to current frame's information
            info_cur_frame.append([cx, cy, box, class_ids[index], kf])
            index += 1

        # every detected object in the first frame is 
        # considered for tracking
        if count == 1:
            for info in info_cur_frame:
                # store the current object's Kalman filter
                curr_kf = info[4]
                
                # updates Kalman filter based on the current object's
                # center point
                curr_kf.update(np.matrix([[info[0]], [info[1]]]))

                # store tracked object's center point, bounding box information, 
                # class name, and Kalman filter object
                tracking_objects[track_id] = info
                tracking_objects[track_id][4] = curr_kf
                track_id += 1

        else:
            tracking_objects_copy = tracking_objects.copy()
            info_cur_frame_copy = info_cur_frame.copy()

            # loop through every detected object in the first frame
            for object_id, info_track in tracking_objects_copy.items():
                object_exists = False
                distances = []
                
                # predict where the object of interest will be 
                # in the current frame (using its Kalman filter)
                curr_kf = info_track[4]
                (x, y) = curr_kf.predict()
                x = x.item((0, 0))
                y = y.item((0, 0))
                pred_cp = (x, y)

                # compute the Euclidean distance between the predicted center 
                # point of object of interest, and the center points
                # every detected object in the current frame
                for info in info_cur_frame_copy:
                    distance = math.hypot(pred_cp[0] - info[0], pred_cp[1] - info[1])
                    distances += [distance]

                # if at least 1 object is detected in the current frame
                if distances != []:
                    # find the minimum Euclidean distance
                    min_dist = min(distances)
                    min_dist_index = distances.index(min_dist)

                    # consider the objects the same, when the minimum Euclidean
                    # distance is less than the distance threshold
                    if min_dist < dist_threshold:
                        info_min_dist = info_cur_frame_copy[min_dist_index]

                        # update Kalman filter based on the center point of 
                        # the object with the minimum distance
                        info = info_min_dist
                        curr_kf.update(np.matrix([[info[0]], [info[1]]]))

                        # store newly tracked object's information
                        tracking_objects[object_id] = info_min_dist
                        tracking_objects[object_id][4] = curr_kf
                        object_exists = True

                        if info_min_dist in info_cur_frame:
                            info_cur_frame.remove(info_min_dist)

                # if the object is no longer present in the current
                # frame, remove it from the list of objects being tracked
                if not object_exists:
                    tracking_objects.pop(object_id)

            # if a new object is detected, add it to the 
            # list of objects being tracked
            for info in info_cur_frame:
                # create a new Kalman filter object
                curr_kf = KalmanFilter(0.1, 1, 1, 1, 0.1,0.1)

                # update Kalman filter based on the center point of 
                # the new object
                curr_kf.update(np.matrix([[info[0]], [info[1]]]))

                # store new object's information
                tracking_objects[track_id] = info
                tracking_objects[track_id][4] = curr_kf
                track_id += 1

        for object_id, info in tracking_objects.items():
            # get the bounding box information for each object that
            # is being tracked in the current frame
            (x, y, w, h) = info[2]

            if show_video == True or save_video == True or measure_success == True:
                # draw the object's bounding box 
                cv2.rectangle(frame, (x, y), (x + w, y + h), (150, 0, 70), 2)

                # write the object label in the format "object ID: class name"
                cv2.rectangle(frame, (x, y - 50), (x + 170, y), (255, 200, 230), -1)
                cv2.putText(frame, f"{str(object_id)}: {class_names[info[3]]}", (x, y - 7), 0, 1, (0, 0, 0), 3, 2)

            if save_info == True:
                # add data to the .csv file
                writer.writerow([count, object_id, class_names[info[3]], info[0], info[1], w, h])

        # copy (store) the center points of the current frame 
        info_prev_frame = info_cur_frame.copy()

        # if desired, show the current frame
        if show_video == True:
            cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
            cv2.imshow("Frame", frame)

        if save_video == True:
            # add the labeled frame to the output video
            output_video.write(frame)

        if measure_success == True:
            # increase the number of pairs of consecutive frames
            # by decreasing the chunk size
            chunk_size = int(frame_count/6)
            if (count % chunk_size == 0 or count % chunk_size == 1) and count > 2:
                # save desired frame as a JPEG file
                cv2. imwrite("images_for_measuring_success/output_frame_%02d.jpg" % file_index, frame)
                file_index += 1
                    
        if show_video == True or save_video == True:
            # press the escape key to close the output video
            key = cv2.waitKey(1)
            if key == 27:
                break
    
    cap.release()

    if show_video == True:
        cv2.destroyAllWindows()
    
    if save_video == True:
        output_video.release()
    
    if save_info == True:
        info_file.close()

# run an example!
object_tracker("20210812_JamesElizabethBrianWithShark.mp4", show_video = False, save_info = True, save_video = True, measure_success = False)