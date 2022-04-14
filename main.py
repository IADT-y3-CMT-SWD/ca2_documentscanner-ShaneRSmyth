'''
LIBRARIES
OpenCV for image processing
Numpy for number matrixes
Datetime for real time
'''
import cv2
import numpy as np
from datetime import datetime


'''
Create trackbars using trackbars function.
Retrieve value from trackbars to increase and
decrease threshold1 and threshold2.
'''
def nothing(x):
    pass

def trackbars(startVal = 125):
    #create window called "Trackbars"
    cv2.namedWindow("Trackbars")
    #resize the window to 360, 240 dimensions
    cv2.resizeWindow("Trackbars", 360, 240)

    #create the two trackbars, Threshold 1 and 2.  Default value is 125.  0-255
    cv2.createTrackbar("Threshold1", "Trackbars", startVal, 255, nothing)
    cv2.createTrackbar("Threshold2", "Trackbars", startVal, 255, nothing)

def value_trackbars():
    #get the values from the trackbars
    try:
        threshold1 = cv2.getTrackbarPos("Threshold1", "Trackbars")
        threshold2 = cv2.getTrackbarPos("Threshold2", "Trackbars")

        src = threshold1, threshold2
        return src
    except:
        #if the trackbars are closed, stop the error and re-open the trackbars
        trackbars()

#find biggest countour
def biggestContour(contours):

    '''
    Loop through list of contours.
    Anything below threshold gets removed.
    Calculate contour perimeter.
    Compare curve with another curve.
    Check if it's a rectangle.
    '''

    biggest = np.array([])
    max_area = 0
    #countours list
    for i in contours:
        area = cv2.contourArea(i)
        #remove below threshold
        if area > 5000:
            #perimeter
            peri = cv2.arcLength(i, True)
            #compare curve with curve
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            # print(f'Area: {area}, Peri: {peri}, Approx: {approx}')
            #check if it's a rectangle
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area

    return biggest, max_area

def reorder(myPoints):
    #reorder the points using numpy
    try:
        myPoints = myPoints.reshape((4, 2))
        myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
        add = myPoints.sum(1)

        myPointsNew[0] = myPoints[np.argmin(add)]
        myPointsNew[3] =myPoints[np.argmax(add)]
        diff = np.diff(myPoints, axis=1)
        myPointsNew[1] =myPoints[np.argmin(diff)]
        myPointsNew[2] = myPoints[np.argmax(diff)]
        
        return myPointsNew
    except ValueError:
        pass   

def drawRectangle(img,biggest,thickness):

    
    #draw rectangle using 4 lines at numpy positions
    try:
        cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 155, 0), thickness)
        cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 155, 0), thickness)
        cv2.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 155, 0), thickness)
        cv2.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 155, 0), thickness)
    
        return img    
    except TypeError:
        pass


########################################################################
#not using webcam so use false
webCamFeed = False
#path to the image being scanned
pathImage = "Images\\image004.jpg"
#default webcam on computer uses 0
cap = cv2.VideoCapture(0)
cap.set(10, 160)
heightImg = 480
widthImg = 640
########################################################################

#activate trackbars
trackbars()
count = 0

while True:
    #webcam or image
    if webCamFeed:
        success, img = cap.read()
    else:
        img = cv2.imread(pathImage)

    #resize the image to height and width dimensions
    img = cv2.resize(img, (heightImg, widthImg))
    #blank image for debugging
    imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)
    #add grayscale to image
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #add gaussian blur to image
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    #get the values from the trackbars
    thres = value_trackbars()
    #stop the error after closing trackbars
    try:
        #add canny blur to image
        imgCanny = cv2.Canny(imgBlur, thres[0], thres[1])
    except TypeError:
        pass
    #kernels needed for dilating and eroding
    kernel = np.ones((5, 5))
    #dilate image
    imgDial = cv2.dilate(imgCanny, kernel, iterations=2)
    #erode image
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)
    #create a copy of original image which will be used for rectangle
    imgContours = img.copy()
    #finding contours on eroded image
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #draw these contours
    cv2.drawContours(imgThreshold, contours, -1, (255, 0, 0), 10)
    

    #get biggest, max_area values from biggestContour using contours
    biggest, max_area = biggestContour(contours)
    #get reordered points using biggest
    myPointsNew = reorder(biggest)
    #draw rectangle on the copied original image, using new points
    draw = drawRectangle(imgContours, myPointsNew, 10)


    '''
    get the matrix values and warp the image
    to read inside the biggest rectangle
    '''
    pts1 = np.float32(myPointsNew)
    pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    #get matrix values using pts1 and pts2
    try:
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
    except cv2.error:
        pass
    #warped image using matrix
    imgWarpColored = cv2.warpPerspective(imgContours, matrix, (widthImg, heightImg))


    '''
    This list displays all images.
    Most are unneeded so commented out.
    Important images are threshold, contours and warped.
    '''

    # cv2.imshow("1. Original", img)
    # cv2.imshow("2. Grayscale", imgGray)
    # cv2.imshow("3. Blur", imgBlur)
    # cv2.imshow("4. Canny", imgCanny)
    # cv2.imshow("5. Dilate", imgDial)
    cv2.imshow("6. Treshold", imgThreshold)
    cv2.imshow("7. imgContours", imgContours)
    cv2.imshow("8. imgWarpColored", imgWarpColored)


    #press x to exit program
    if cv2.waitKey(1) & 0XFF == ord('x'):
        break

    #press s to save image
    if cv2.waitKey(1) & 0xFF == ord('s'):
        print("saving")

        while True:
            #ask user if they want a watermark on the image
            watermarkOption = input("Do you want to add a watermark to the image? y/n ")
            while True:
                #if the user pressed y
                if watermarkOption == "y":
                    print("Adding watermark to image.")
                    watermark = "CMTY3"
                    #put watermark on warped image.         #position   #font                 #scale #colour #thickness
                    cv2.putText(imgWarpColored, watermark, (45, 375), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 10)

                    #break out of loop to next question
                    break
                #if the user presses n
                elif watermarkOption == "n":
                    
                    #break out of loop to next question without adding watermark
                    break
            
            #ask user if they want a timestamp on the image
            timestampOption = input("Do you want to add a timestamp to the image? y/n ")
            while True:
                #if the user entered y
                if timestampOption == "y":
                    print("Adding timestamp to image.")
                    #get the current time and date
                    timestamp = datetime.now()
                    #put the timestamp on warped image as a string
                    cv2.putText(imgWarpColored, str(timestamp), (45, 450), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 5)

                    #then break out of loop
                    break
                
                #if user entered n
                elif timestampOption == "n":
                    
                    #break ouf of loop without putting timestamp on image
                    break
            #finished asking questions, break out of parent loop
            break
                    
        #save warped image to Scanned folder and increase count every time s is pressed
        cv2.imwrite("Scanned/myImage"+str(count)+".jpg", imgWarpColored)
        cv2.waitKey(300)
        count += 1

#release the video capture
cap.release()

#close everything
cv2.destroyAllWindows()
