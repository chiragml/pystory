#!/user/bin/env python
#!pip install pybase64
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
import argparse

def main(csv_file_path, video_path, save_path):
    spark = SparkSession.builder.appName("preprocessing-data").config("spark.memory.offHeap.enabled","true").config("spark.memory.offHeap.size","10g").getOrCreate()

    # Loading the csv
    df_csv = spark.read.csv(csv_file_path, header=True)

    video_dir = "/data/video/video/"
    video_path = video_path
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
        success, first_frame = vidcap.read()
        if not success:
            return None, None
        frame_count = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
        success, last_frame = vidcap.read()
        if not success:
            return None, None
        save_img = first_frame
        base64_img1 = cv2.imencode('.jpg', first_frame)[1]
        base64_img1 = base64_img1.tobytes()
        im64_1 = pybase64.b64encode(base64_img1).decode('utf-8')
        base64_img2 = cv2.imencode('.jpg', last_frame)[1]
        base64_img2 = base64_img2.tobytes()
        im64_2 = pybase64.b64encode(base64_img2).decode('utf-8')
        vidcap.release()
        return [im64_1, im64_2]

    get_frame_udf = udf(get_frame)

    # Get the first frame of the video and save it as an image (this step preferrably in parallel)
    
    video_df = video_df.withColumn('images', get_frame_udf(video_df['video_path']))
    # video_df = video_df.withColumn('last_frame', get_frame_udf(video_df['video_path'])[1])
    print(video_df.head(3))
    final_df = video_df.join(df_csv, video_df.video_name == df_csv.video, 'inner')
    # print(final_df.head(3))
    final_df = final_df.write.csv(save_path + "data.csv", header=True, mode="overwrite")
    # final_df.save.csv('final_df.csv', header=True, mode='overwrite')

    spark.stop()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess the data')
    parser.add_argument('--csv_file_path', type=str, help='The path to the csv file')
    parser.add_argument('--video_path', type=str, help='The path to the video files')
    parser.add_argument('--save_path', type=str, help='The path to save the data')
    args = parser.parse_args()
    main(args.csv_file_path, args.video_path, args.save_path)
    # main('/data/video/video.csv', '/data/video/video/', '/data/video/')