import cv2
import numpy as np
import os
from os import listdir

dir = 'Directory to Video Folder'
vidFolder = os.listdir(dir)

max_frame = 300

vidSequence =[]

i = 0

for video_path in vidFolder:
    video_path = dir + "/" + video_path
    video = cv2.VideoCapture(video_path)

    frames = []
    
    while True:
        ret, frame = video.read()
        
        if not ret:
            break
    
        frames.append(frame)
        
    video.release()

    num_frames = len(frames)
    padded_frames = []
    if num_frames < max_frame:
        padding_frames = [frames[-1]] * (max_frame - num_frames)
        padded_frames = frames + padding_frames
    else:
        padded_frames = frames[:max_frame]

    resized_frames = []
    for frame in padded_frames:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        resized_frame = cv2.resize(gray_frame, (224, 224)) 
        resized_frames.append(resized_frame)
    
    vidSequence.append(resized_frames)

    i += 1
    print(i)

print(np.shape(vidSequence))

np.save('vidData.npy', vidSequence)

print('end')