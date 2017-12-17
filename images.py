import glob
from pipeline import *
from lane_finder import *

pipeline = ImagePipeline()

images = glob.glob('./camera_cal/*.jpg')
for fname in images:
    filename = fname.split('/')[-1]
    img = cv2.imread(fname)
    proc = pipeline.undistort(img)
    cv2.imwrite("./output_images/camera_cal/" + filename, proc)

images = glob.glob('./test_images/*.jpg')
for fname in images:
    filename = fname.split('/')[-1]

    img = cv2.imread(fname)
    undist = pipeline.undistort(img)
    cv2.imwrite("./output_images/01-undistort/" + filename, undist)
    undist = cv2.cvtColor(undist, cv2.COLOR_BGR2RGB)

    proc = cv2.cvtColor(undist, cv2.COLOR_RGB2HLS)
    proc = pipeline.find_lanes(proc)
    binary = np.array(proc, dtype=np.uint8)
    binary = np.dstack((binary, binary, binary)) * 255
    cv2.imwrite("./output_images/02-binary/" + filename, binary)

    lines = pipeline.aoi.get_lines()
    und_lines = np.copy(undist)
    for line in lines:
        cv2.line(und_lines, (int(line[0]), int(line[1])), (int(line[2]), int(line[3])), (255, 0, 0), 2)
    und_lines = cv2.cvtColor(und_lines, cv2.COLOR_RGB2BGR)
    cv2.imwrite("./output_images/03.1-warp/original/" + filename, und_lines)

    warped = pipeline.persp.warp(und_lines)
    cv2.imwrite("./output_images/03.1-warp/warped-original/" + filename, warped)

    warped = pipeline.persp.warp(binary)
    cv2.imwrite("./output_images/03.1-warp/warped-binary/" + filename, warped)

    finder = LaneFinder(1)
    proc = pipeline.process_frame(undist)
    finder.find_lanes(proc)
    lanes = finder.mark_lane(proc)
    cv2.imwrite("./output_images/04-lanes/" + filename, lanes)

    proc = pipeline.unwarp_and_combine(cv2.cvtColor(undist, cv2.COLOR_BGR2RGB), lanes)
    cv2.imwrite("./output_images/05-unwarped/" + filename, proc)

    proc = finder.annotate_frame(proc)
    cv2.imwrite("./output_images/" + filename, proc)
