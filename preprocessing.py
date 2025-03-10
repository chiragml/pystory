'''
-------------------------------
Description
-------------------------------
This file contains the preprocessing functions for the data.
The data needs to be processed in parallel and set into the following format:
videl_name, First_frame, description, video_path

Using pyspark to process the data in parallel on GCP's Dataproc- (hopefully for free but lets see)
-------------------------------
'''
import os
import pandas as pd
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql.functions import *
from pyspark.sql.functions import col, udf
import pathlib
import cv2
import glob
from PIL import Image
import pybase64

spark = SparkSession.builder.appName("preprocessing-data").config("spark.memory.offHeap.enabled","true").config("spark.memory.offHeap.size","10g").getOrCreate()

# Loading the csv
df_csv = spark.read.csv(os.getcwd() + "/data/OpenVid-1M.csv", header=True)

video_dir = "/data/video/video/"
video_path = os.getcwd() + video_dir
save_path = "./data/"
# Getting the video names
videos = glob.glob(video_path + '*.mp4')
# videos = pathlib.Path(video_path).glob('*.mp4 ')
video_list = [str(vids) for vids in videos]

# Getting the video names
video_names = [vid.split('/')[-1] for vid in video_list]
# print(video_list[0:10])

# Generate a pyspark dataframe with video names and video list as columns
video_df = pd.DataFrame({'video_name': video_names, 'video_path': video_list})
video_df = spark.createDataFrame(video_df)

# Now need take the video and get the first frame and save that as image in the df

'''
-------------------------------
Description
-------------------------------
Function to get the first frame of the video and save it as an image.

-------------------------------
Parameters
-------------------------------
video_path: str
    The path to the video file

-------------------------------
Returns
-------------------------------
image: np.array
    The first frame of the video

-------------------------------
'''
def get_frame(video_path):
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    if not success:
        return None
    save_img = image
    base64_img = cv2.imencode('.jpg', image)[1]
    base64_img = base64_img.tobytes()
    im64 = pybase64.b64encode(base64_img).decode('utf-8')
    vidcap.release()
    return im64

get_frame_udf = udf(get_frame)

# Get the first frame of the video and save it as an image (this step preferrably in parallel)
video_df = video_df.withColumn('image', get_frame_udf(video_df['video_path']))
# print(video_df.head(3))
final_df = video_df.join(df_csv, video_df.video_name == df_csv.video, 'inner')
# print(final_df.head(3))
final_df = final_df.write.csv(save_path + "data.csv", header=True, mode="overwrite")
# final_df.save.csv('final_df.csv', header=True, mode='overwrite')

spark.stop()