'''
libraries
'''

import cv2
import numpy as np


def nothing(x):
    pass

def trackbars(startVal = 125):
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 360, 240)

    cv2.createTrackbar("Threshold1", "Trackbars", startVal, 255, nothing)
    cv2.createTrackbar("Threshold2", "Trackbars", startVal, 255, nothing)

def value_trackbars():
    threshold1 = cv2.getTrackbarPos("Threshold1", "Trackbars")
    threshold2 = cv2.getTrackbarPos("Threshold2", "Trackbars")

    src = threshold1, threshold2
    return src

#find biggest countour
def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    #loop through countours list
    for i in contours:
        area = cv2.contourArea(i)
        #discard areas below treshold
        if area > 5000:
            #calculates a contour perimeter -> float
            peri = cv2.arcLength(i, True)
            #calcualtes a curve or a polygon with another curve/polygon with less vertices
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            # print(f'Area: {area}, Peri: {peri}, Approx: {approx}')
            # is it a rectangle?
            if area > max_area and len(approx) == 4:
                biggest = approx
                #overwrite max_area for regions that are lareger
                max_area = area
    return biggest, max_area

#reordering points in order to warp image
def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)

    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] =myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] =myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    
    return myPointsNew   
 
def drawRectangle(img,biggest,thickness):
    cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 155, 0), thickness)
    cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 155, 0), thickness)
    cv2.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 155, 0), thickness)
    cv2.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 155, 0), thickness)
 
    return img

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
    thres = value_trackbars()  # GET TRACK BAR VALUES FOR THRESHOLDS
    imgCanny = cv2.Canny(imgBlur, thres[0], thres[1]) #thres[0], thres[1])  # APPLY CANNY BLUR
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=2)  # APPLY DILATION
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)  # APPLY EROSION 

    imgContours = img.copy()
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(imgThreshold, contours, -1, (255, 0, 0), 10)
    
    biggest, max_area = biggestContour(contours)

    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

    biggestContour(contours)
    reorder(biggest)
    myPointsNew = reorder(biggest)
    drawRectangle(imgContours, myPointsNew, 10)


    cv2.imshow("1. Original", img)
    cv2.imshow("2. Grayscale", imgGray)
    cv2.imshow("3. Blur", imgBlur)
    cv2.imshow("4. Canny", imgCanny)
    cv2.imshow("5. Dilate", imgDial)
    cv2.imshow("6. Treshold", imgThreshold)
    cv2.imshow("7. imgContours", imgContours)

    # imgOutput = imgCanny

    # Press x  on keyboard to  exit
    # Close and break the loop after pressing "x" key
    if cv2.waitKey(1) & 0XFF == ord('x'):
        break  # exit infinite loop

    # SAVE IMAGE WHEN 's' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('s'):
        print("saving")
        # save image to folder using cv2.imwrite()
        cv2.imwrite("Scanned/myImage"+str(count)+".jpg", imgContours)
        cv2.waitKey(300)
        count += 1

# When everything done, release
# the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
