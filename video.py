import glob
from moviepy.editor import VideoFileClip
from lane_finder import *
from pipeline import *

pipeline = ImagePipeline()


def proc_frame(img):
    und = pipeline.undistort(img)
    proc = pipeline.process_frame(und)
    finder.find_lanes(proc)
    proc = finder.mark_lane(proc)
    proc = pipeline.unwarp_and_combine(und, proc)
    proc = finder.annotate_frame(proc)
    return proc


videos = glob.glob("./*.mp4")
for video in videos:
    finder = LaneFinder(5)
    filename = video.split('/')[-1]
    clip1 = VideoFileClip(video)
    proc_clip = clip1.fl_image(proc_frame)
    proc_clip.write_videofile("output_video/" + filename, audio=False)
