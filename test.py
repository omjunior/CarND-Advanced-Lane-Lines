import glob
from moviepy.editor import VideoFileClip
from lane_finder import *
from pipeline import *

pipeline = ImagePipeline()
finder = LaneFinder(5)

# images = glob.glob('./test_images/*.jpg')
# for fname in images:
#     img = cv2.imread(fname)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     und = pipeline.undistort(img)
#     proc = pipeline.process_frame(und)
#     finder.find_lanes(proc)
#     proc = finder.mark_lane(proc)
#     proc = pipeline.unwarp_and_combine(cv2.cvtColor(und, cv2.COLOR_RGB2BGR), proc)
#     proc = finder.annotate_frame(proc)
#     filename = fname.split('/')[-1]
#     cv2.imwrite("./output_images/" + filename, proc)


def proc_frame(img):
    und = pipeline.undistort(img)
    proc = pipeline.process_frame(und)
    finder.find_lanes(proc)
    proc = finder.mark_lane(proc)
    proc = pipeline.unwarp_and_combine(und, proc)
    proc = finder.annotate_frame(proc)
    return proc


clip1 = VideoFileClip("project_video.mp4")
proc_clip = clip1.fl_image(proc_frame)
proc_clip.write_videofile("output_video/project_video_out.mp4", audio=False)
