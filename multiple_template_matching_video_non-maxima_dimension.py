# Importing libraries
from imutils.object_detection import non_max_suppression
import argparse
import numpy as np
import cv2

# Making threshold value as an argument to pyhton file
ap = argparse.ArgumentParser()
ap.add_argument("-b", "--threshold", type=float, default=0.8,
    help="threshold for multi-template matching")
args = vars(ap.parse_args())

cam = cv2.VideoCapture(0)

while(1):

    # Reading images
    template = cv2.imread('images/temp.png')
    _, image = cam.read()
    (tH, tW) = template.shape[:2]

    # Load Aruco detector
    parameters = cv2.aruco.DetectorParameters_create()
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)
    corners, _, _ = cv2.aruco.detectMarkers(image, aruco_dict, parameters=parameters)
    # Draw polygon around the marker
    int_corners = np.int0(corners)
    cv2.polylines(image, int_corners, True, (0, 255, 0), 5)

    # Aruco Perimeter
    if (len(corners)!=0):
        aruco_perimeter = cv2.arcLength(corners[0], True)
    # Pixel to cm ratio
        pixel_cm_ratio = aruco_perimeter / 20 
    else:
        pixel_cm_ratio=60

    # Converting BGR to GRAY scale images
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # Implementing template matching
    result = cv2.matchTemplate(imageGray, templateGray,
    cv2.TM_CCOEFF_NORMED)

    # Finding coordinates which are greater than threshold
    (yCoords, xCoords) = np.where(result >= args["threshold"])

    # Implementing non-maxima supression to remove duplicated images
    rects = []
    # Filling array 
    for (x, y) in zip(xCoords, yCoords):
        rects.append((x, y, x + tW, y + tH))
    # Applying non-maxima
    pick = non_max_suppression(np.array(rects))
    print("[INFO] {} matched cards".format(len(pick)))
    matched_cards = len(pick)

    # Drawing rectangles
    for (startX, startY, endX, endY) in pick:
        cv2.rectangle(image, (startX, startY), (endX, endY),
            (255, 0, 0), 3)
        object_width = endX / pixel_cm_ratio
        object_height = endY / pixel_cm_ratio
        cv2.putText(image, "{}".format(round(object_width, 1)), (int(startX), int(startY)), cv2.FONT_HERSHEY_PLAIN, 1.5, (100, 200, 0), 1)
        cv2.putText(image, "{}".format(round(object_height, 1)), (int(startX), int(startY+20)), cv2.FONT_HERSHEY_PLAIN, 1.5, (100, 200, 0), 1)


    x, y = 25, 25
    cv2.putText(image, f"Number of cards: {matched_cards}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))

    # Show result
    cv2.imshow("Result", image)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows() 
        break
