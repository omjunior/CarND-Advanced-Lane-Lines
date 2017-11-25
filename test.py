import cv2
import numpy as np
import glob
from camera import Camera
from perspective import PerspectiveTransformer
from image import draw_lines

camera = Camera("camera_cal/calibration.p")

img = cv2.imread("test_images/straight_lines1.jpg")
img_shape = img.shape[1::-1]
bottom_width = 1.0
bottom_y_pos = 0.95
top_width = 0.185
top_y_pos = 0.65
src = [[ img_shape[0]/2 - (img_shape[0]*bottom_width)/2, img_shape[1]*bottom_y_pos ], \
       [ img_shape[0]/2 + (img_shape[0]*bottom_width)/2, img_shape[1]*bottom_y_pos ], \
       [ img_shape[0]/2 - (img_shape[0]*top_width)/2, img_shape[1]*top_y_pos ], \
       [ img_shape[0]/2 + (img_shape[0]*top_width)/2, img_shape[1]*top_y_pos ]]
print(src)
src_lines = [[src[0][0], src[0][1], src[1][0], src[1][1]], \
             [src[1][0], src[1][1], src[3][0], src[3][1]], \
             [src[3][0], src[3][1], src[2][0], src[2][1]], \
             [src[2][0], src[2][1], src[0][0], src[0][1]]]
src_lines = np.array(src_lines, dtype=np.uint32)

dst = [[0, img_shape[1]], [img_shape[0], img_shape[1]], [0, 0], [img_shape[0], 0]]
print(dst)
transform = PerspectiveTransformer(np.array(src, dtype=np.float32), np.array(dst, dtype=np.float32))

images = glob.glob('test_images/*.jpg')
for fname in images:
    img = cv2.imread(fname)
    undist = camera.undistort(img)
    draw_lines(undist, src_lines)
    only_file = fname.split('/')[-1].split('\\')[-1]
    cv2.imwrite("output_images/"+only_file+"_undist.jpg", undist)
    persp = transform.transform(undist)
    cv2.imwrite("output_images/"+only_file+"_birdseye.jpg", persp)
