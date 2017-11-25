import cv2

"""
Image helper functions
"""

def draw_lines(img, lines, color=[0, 0, 255], thickness=2):
    """ Draw lines over an image """
    for line in lines:
        x1,y1,x2,y2 = line
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)
