import cv2
import numpy as np

# video data 
import os
from os import listdir

dir = 'Your Directory'
vidFolder = os.listdir(dir)

max_frame = 250

vidSequence =[]

for video_path in vidFolder:
    video = cv2.VideoCapture(video_path)

    frames = []
    
    # Read the video frame by frame
    while video.isOpened():
        ret, frame = video.read()
        
        # when frame is not read 
        if not ret:
            print('frame error')
            break

    # Release the video object to avoid memory leaks
    video.release()

    # Perform padding if necessary
    num_frames = len(frames)
    padded_frames = []
    if num_frames < max_frame:
        padding_frames = [frames[-1]] * (max_frame - num_frames)
        padded_frames = frames + padding_frames
    else:
        padded_frames = frames[:max_frame]

    # Resize frames if necessary
    resized_frames = []
    for frame in padded_frames:
        resized_frame = cv2.resize(frame, (224, 224))
        resized_frames.append(resized_frame)
    
    vidSequence.append(resized_frames)

print(vidSequence[0])

np.save('vidSequence', vidSequence)