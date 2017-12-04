import cv2
import numpy as np
import glob

from lane_finder import *
from pipeline import *


pipeline = ImagePipeline()
finder = LaneFinder(1280, 0)

# images = glob.glob('test_images/*.jpg')
# for fname in images:
#     img = cv2.imread(fname)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     proc = pipeline.process_frame(img)
#     finder.find_lanes(proc)
#     proc = finder.paint(proc)
#     filename = fname.split('/')[-1].split('\\')[-1]
#     cv2.imwrite("output_images/"+filename, proc)

def proc_frame(img):
    proc = pipeline.process_frame(img)
    finder.find_lanes(proc)
    proc = finder.paint(proc)
    return proc

from moviepy.editor import VideoFileClip
clip1 = VideoFileClip("project_video.mp4")
proc_clip = clip1.fl_image(proc_frame)
proc_clip.write_videofile("output_video/project_video_out.mp4", audio=False)
