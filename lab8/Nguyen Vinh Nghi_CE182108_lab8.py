# Nguyen Vinh Nghi 
# CE182108

import cv2
import numpy as np
import matplotlib.pyplot as plt

v = cv2.VideoCapture('./video.mp4')

option = 'meanshift'


frame_width = int(v.get(3))
frame_height = int(v.get(4))
size = (frame_width, frame_height)

result = cv2.VideoWriter('./output.mp4',  
                         cv2.VideoWriter_fourcc(*'MP4V'), 
                         60, size) 

trajectory_points = [] 

if __name__ == '__main__':
    ret, frame = v.read()

    bbox = [300, 400, 200, 200]

    roi = frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
    cv2.imshow('Camera', roi)

    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))

    hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])

    norm_hist = cv2.normalize(hist, None, 0, 255, cv2.NORM_MINMAX)
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    while True:
        ret, frame = v.read()
        if not ret or cv2.waitKey(1) == ord('q'):
            break
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], norm_hist, [0, 180], 1)

        if option == 'meanshift':
            ret, bbox = cv2.meanShift(dst, bbox, term_crit)
        if option == 'camshift':
            ret, bbox = cv2.CamShift(dst, bbox, term_crit)

        center = (int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2))
        trajectory_points.append(center)

        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 0, 0), 2)
        for i in range(1, len(trajectory_points)):
            cv2.line(frame, trajectory_points[i - 1], trajectory_points[i], (0, 255, 0), 2)  # Draw green lines

        result.write(frame)
        cv2.imshow('Camera', frame)
        
    v.release()
    result.release()
    cv2.destroyAllWindows()
