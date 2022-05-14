# Shark-and-Human-Tracking
**CSCI 153 Final Project** <br>
For my final project, I have created a program that can identify and track sharks and humans in consecutive frames of a drone video sequence. This repository contains files that are necessary to run this program. <br>

To use the program, install Python and OpenCV. <br>

Then, download the following files:
1. classes.txt
2. yolov4-obj.cfg
3. object_detection.py
4. kalman_filter.py
5. object_tracking.py

Next, download the training weights (yolov4-obj_final.weights) here: https://drive.google.com/file/d/14dqXPAmrkqzYu21CO5kzlUSSkQslNcrM/view?usp=sharing

Before running the program, a little bit of file organization is necessary:
1. Create a folder named "dnn_model" in the same folder that you have downloaded the 5 files.
2. Move the yolov4-obj.cfg, yolov4-obj_final.weights, and classes.txt files to the "dnn_model" folder.

### Object Detection
The object detector is created using **object_detection.py**, which was obtained from an Pysource tutorial. It can be downloaded from the original website using the following link: https://pysource.com/wp-content/uploads/2021/10/Object-tracking-from-scratch-source_code.zip. However, I recommend using the object_detection.py file in this repository, as it is configured to work with the file organization steps (listed above).

### Kalman Filtering
Kalman filtering is implemented using **kalman_filter.py**, which was obtained from a repository by GitHub user RahmadSadli. The original code (KalmanFilter.py) and GitHub repository for the Kalman filter class can be found here: https://github.com/RahmadSadli/2-D-Kalman-Filter. Again, I recommend using the kalman_filter.py file in this repository, as it is configured to work the object tracking file (described below). 

### Object Tracking
To run the shark and human tracking program, which calls the object detector and Kalman filter classes explained above, use **object_tracking.py**. This file was adapted from starter code provided in the same Pysource tutorial as object_detection.py. However, I have modified it to use Kalman filtering and formatted it as a function that can be called.

The function is called object_tracker(), and it takes a drone video and four booleans as its input. These booleans allow users to indicate which type(s) of output they would like. The four boolean arguments are listed and described below:
1. *show_video*: opens a new window that displays each labeled video frame (one-by-one).
2. *save_info*: creates and saves a CSV file containing the object ID, class name, center point coordinates (x-coordinate and y-coordinate), and bounding box dimensions (width and height) for each detected object in each frame.
3. *save_video*: creates and saves an AVI file of the labeled video, by putting the labeled frames together and using the same frame rate as that of the input video.
4. *measure_success*: obtains and saves 6 pairs of consecutive frames from the input video.

The function gets called in the last line of object_tracking.py (Line 233). Here, you can set each argument to True/False, depending on which type(s) of output you would like. You will also need to add a drone video, as the function's first argument. Below is an example call of the function:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; object_tracker("20210812_JamesWithShark.mp4", show_video = False, save_info = True, save_video = True, measure_success = False)

Note that the video needs to be in the same folder as the object_tracking.py file!

Once you've customized the function call, run object_tracking.py in the terminal. 
