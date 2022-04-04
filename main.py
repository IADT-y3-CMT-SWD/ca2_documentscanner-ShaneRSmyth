'''
libraries
'''

import cv2
import numpy as np

def nothing():
    pass

def trackbars(startVal = 125):
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 360, 240)
    cv2.createTrackbar("Threshold1", "Trackbars", startVal, 255, nothing)


########################################################################
webCamFeed = False  # set to false if no webcam available
pathImage = "Images\\image004.jpg"
# main webcam -> 0
cap = cv2.VideoCapture(0)
cap.set(10, 160)
heightImg = 480
widthImg = 640
########################################################################

trackbars()
count = 0

while True:
    # input is either webcam or image
    if webCamFeed:
        success, img = cap.read()
    else:
        img = cv2.imread(pathImage)

    # RESIZE IMAGE
    img = cv2.resize(img, (heightImg, widthImg))
    # CREATE A BLANK IMAGE FOR TESTING DEBUGING IF REQUIRED
    imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)
    # CONVERT IMAGE TO GRAY SCALE
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  # ADD GAUSSIAN BLUR
    # thres = valTrackbars()  # GET TRACK BAR VALUES FOR THRESHOLDS
    imgCanny = cv2.Canny(imgBlur, 150, 100) #thres[0], thres[1])  # APPLY CANNY BLUR
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=2)  # APPLY DILATION
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)  # APPLY EROSION
    
    imgContours = img.copy()
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(imgThreshold, contours, -1, (255, 0, 0)), 2

    cv2.imshow("1. Original", img)
    cv2.imshow("2. Grayscale", imgGray)
    cv2.imshow("3. Blur", imgBlur)
    cv2.imshow("4. Canny", imgCanny)
    cv2.imshow("5. Dilate", imgDial)
    cv2.imshow("6. Treshold", imgThreshold)
    cv2.imshow("7. imgContours", imgContours)

    imgOutput = imgCanny

    # Press x  on keyboard to  exit
    # Close and break the loop after pressing "x" key
    if cv2.waitKey(1) & 0XFF == ord('x'):
        break  # exit infinite loop

     # SAVE IMAGE WHEN 's' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('s'):
        print("saving")
        # save image to folder using cv2.imwrite()
        cv2.imwrite("Scanned/myImage"+str(count)+".jpg", imgOutput)
        cv2.waitKey(300)
        count += 1

# When everything done, release
# the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
