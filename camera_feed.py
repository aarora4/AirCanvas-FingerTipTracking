import cv2
import math
from keras.models import load_model
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# ================================================================= Code for Emotion Mapping =================================================================

# train_dir = 'data/train'
# val_dir = 'data/test'
# train_datagen = ImageDataGenerator(rescale=1./255)
# val_datagen = ImageDataGenerator(rescale=1./255)

# train_generator = train_datagen.flow_from_directory(
#         train_dir,
#         target_size=(48,48),
#         batch_size=64,
#         color_mode="grayscale",
#         class_mode='categorical')

# validation_generator = val_datagen.flow_from_directory(
#         val_dir,
#         target_size=(48,48),
#         batch_size=64,
#         color_mode="grayscale",
#         class_mode='categorical')

# emotion_model = Sequential()

# emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
# emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
# emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
# emotion_model.add(Dropout(0.25))

# emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
# emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
# emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
# emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
# emotion_model.add(Dropout(0.25))

# emotion_model.add(Flatten())
# emotion_model.add(Dense(1024, activation='relu'))
# emotion_model.add(Dropout(0.5))
# emotion_model.add(Dense(7, activation='softmax'))
# # emotion_model.load_weights('emotion_model.h5')

# cv2.ocl.setUseOpenCL(False)

# emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}


# emotion_model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001, decay=1e-6),metrics=['accuracy'])
# emotion_model_info = emotion_model.fit_generator(
#         train_generator,
#         steps_per_epoch=28709 // 64,
#         epochs=50,
#         validation_data=validation_generator,
#         validation_steps=7178 // 64)
# emotion_model.save_weights('emotion_model.h5')

# ================================================================= Code for Hand Pose Estimation with CMU Model =================================================================

# prototypeFile = "hand/pose_deploy.prototxt"
# handPoseModel = "hand/pose_iter_102000.caffemodel"

# nPoints = 22
# net = cv2.dnn.readNetFromCaffe(prototypeFile, handPoseModel)
# threshold = 0.25
gesture = []

cap = cv2.VideoCapture(0)

total_rectangle = 9

# rect_size = 20

rows, cols = 1280, 720

hand_rect_one_x = np.array(
        [6 * rows / 20, 6 * rows / 20, 6 * rows / 20, 9 * rows / 20, 9 * rows / 20, 9 * rows / 20, 12 * rows / 20,
         12 * rows / 20, 12 * rows / 20], dtype=np.uint32)

hand_rect_one_y = np.array(
    [9 * cols / 20, 10 * cols / 20, 11 * cols / 20, 9 * cols / 20, 10 * cols / 20, 11 * cols / 20, 9 * cols / 20,
        10 * cols / 20, 11 * cols / 20], dtype=np.uint32)

hand_rect_two_x = hand_rect_one_x + 10
hand_rect_two_y = hand_rect_one_y + 10

# emotion_map = {0:"Angry", 1:"Disgust", 2:"Fear", 3:"Happy", 4:"Neutral", 5:"Sad", 6:"Surprise"}

# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

histogram_made = False
frame_wait = 25

bgSubtractor = None

rolling_count = np.zeros([10])

hand_histogram = None

distance_from_centroid_to_max = []

def rolling_finger_count(currInt):
    global rolling_count
    if rolling_count.size < 10:
        rolling_count = np.append(rolling_count, currInt)
        
    else:
        rolling_count = np.insert(rolling_count, rolling_count.size, currInt)
        rolling_count = np.delete(rolling_count, 0)
        
    
    # print(rolling_count)
    return np.mean(rolling_count)

def bgSubMasking(frame):
    """Create a foreground (hand) mask
    @param frame: The video frame
    @return: A masked frame
    """
    fgmask = bgSubtractor.apply(frame, learningRate=0)    

    kernel = np.ones((4, 4), np.uint8)
    
    # The effect is to remove the noise in the background
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=2)
    # To close the holes in the objects
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Apply the mask on the frame and return
    return cv2.bitwise_and(frame, frame, mask=fgmask)

def hist_masking(frame, hist):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)

    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    cv2.filter2D(dst, -1, disc, dst)

    ret, thresh = cv2.threshold(dst, 150, 255, cv2.THRESH_BINARY)

    # thresh = cv2.dilate(thresh, None, iterations=5)
    kernel = np.ones((3, 3), np.uint8)
    # thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    # To close the holes in the objects
    # thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=9)
    thresh = cv2.merge((thresh, thresh, thresh))

    # return thresh

    return cv2.bitwise_and(frame, thresh)

def threshold(mask):
    """Thresholding into a binary mask"""
    grayMask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grayMask, (9, 9), 0)
    ret, thresh = cv2.threshold(blur, 0, 255, 0)
    return thresh

def getMaxContours(contours):
    """Find the largest contour"""
    maxIndex = 0
    maxArea = 0
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        if area > maxArea:
            maxArea = area
            maxIndex = i
    return contours[maxIndex]

def rescale_frame(frame, wpercent=130, hpercent=130):
    width = int(frame.shape[1] * wpercent / 100)
    height = int(frame.shape[0] * hpercent / 100)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

def draw_rect(frame):
    rows, cols, _ = frame.shape
    global total_rectangle, hand_rect_one_x, hand_rect_one_y, hand_rect_two_x, hand_rect_two_y

    hand_rect_one_x = np.array(
        [6 * rows / 20, 6 * rows / 20, 6 * rows / 20, 9 * rows / 20, 9 * rows / 20, 9 * rows / 20, 12 * rows / 20,
         12 * rows / 20, 12 * rows / 20], dtype=np.uint32)

    hand_rect_one_y = np.array(
        [9 * cols / 20, 10 * cols / 20, 11 * cols / 20, 9 * cols / 20, 10 * cols / 20, 11 * cols / 20, 9 * cols / 20,
         10 * cols / 20, 11 * cols / 20], dtype=np.uint32)

    hand_rect_two_x = hand_rect_one_x + 10
    hand_rect_two_y = hand_rect_one_y + 10

    for i in range(total_rectangle):
        cv2.rectangle(frame, (hand_rect_one_y[i], hand_rect_one_x[i]),
                      (hand_rect_two_y[i], hand_rect_two_x[i]),
                      (0, 255, 0), 1)

    return frame

def find_centroid(max_contour, frame):
    M = cv2.moments(max_contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    cv2.circle(frame, (cX, cY), 7, (255, 255, 255), -1)

    return frame

def find_centroid_point(max_contour):
    M = cv2.moments(max_contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    return (cX, cY)


def countFingers(contour):
    hull = cv2.convexHull(contour, returnPoints=False)
    if len(hull) > 3:
        defects = cv2.convexityDefects(contour, hull)
        cnt = 0
        if type(defects) != type(None):
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(contour[s, 0])
                end = tuple(contour[e, 0])
                far = tuple(contour[f, 0])
                angle = calculateAngle(far, start, end)
                
                # Ignore the defects which are small and wide
                # Probably not fingers
                if d > 10000 and angle <= math.pi/2:
                    cnt += 1
        if cnt == 0:
            centroid = find_centroid_point(contour)
            highest_point = tuple(contour[contour[:, :, 1].argmin()][0])
            x1, y1 = centroid
            x2, y2 = highest_point
            lowest_point = tuple(contour[contour[:, :, 1].argmax()][0])
            dist_a = np.linalg.norm(np.asarray(centroid) - np.asarray(highest_point))
            dist_b = np.linalg.norm(np.asarray(highest_point) - np.asarray(lowest_point))
            if (dist_a / dist_b) > 0.525:
                cnt = -1

        return True, cnt
    return False, 0
    
def calculateAngle(far, start, end):
    """Cosine rule"""
    a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
    b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
    c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
    angle = math.acos((b**2 + c**2 - a**2) / (2*b*c))
    return angle

def hand_histogram(frame):
    global hand_rect_one_x, hand_rect_one_y

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    roi = np.zeros([90, 10, 3], dtype=hsv_frame.dtype)

    for i in range(total_rectangle):
        roi[i * 10: i * 10 + 10, 0: 10] = hsv_frame[hand_rect_one_x[i]:hand_rect_one_x[i] + 10,
                                          hand_rect_one_y[i]:hand_rect_one_y[i] + 10]

    hand_hist = cv2.calcHist([roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
    return cv2.normalize(hand_hist, hand_hist, 0, 255, cv2.NORM_MINMAX)

# def adjust_gamma(image, gamma=1.0):

#    invGamma = 1.0 / gamma
#    table = np.array([((i / 255.0) ** invGamma) * 255
#       for i in np.arange(0, 256)]).astype("uint8")

#    return cv2.LUT(image, table)
# fgbg = cv2.createBackgroundSubtractorMOG2()

contour_mode = False

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    # bgMask = fgbg.apply(frame)
    # frame = adjust_gamma(frame, 0.9)
    # print(frame.shape)
    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    # height, width = frame.shape[:2]
    # max_height = 48
    # max_width = 48

    # shrunk = gray

    # ================================================================= Code for Face Blurrings =================================================================

    # faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # for (x, y, w, h) in faces:
        # cv2.rectangle(gray, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # region_of_interest = gray[y:y+h, x:x+w]
        # region_of_interest = cv2.GaussianBlur(region_of_interest, (51, 51), cv2.BORDER_DEFAULT)
        # region_of_interest = cv2.GaussianBlur(region_of_interest, (51, 51), cv2.BORDER_DEFAULT)
        # gray[y:y+h, x:x+w] = region_of_interest

    # ================================================================= Code for Emotion Mapping =================================================================
    # # only shrink if img is bigger than required
    # if max_height < height or max_width < width:
    #     # get scaling factor
    #     scaling_factor = max_height / float(height)
    #     if max_width/float(width) < scaling_factor:
    #         scaling_factor = max_width / float(width)
    #     # resize image
    # shrunk = np.expand_dims(np.expand_dims(cv2.resize(gray, (48, 48)), -1), 0)

    # print(shrunk.shape)
    # res = emotion_model.predict(shrunk)
    # maxindex = int(np.argmax(res))
    # cv2.putText(gray, emotion_map[maxindex], (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # shrunk = cv2.resize(gray, (int(frame.shape[0] / 15), int(frame.shape[1] / 15)))

    # ================================================================= Code for Hand Pose Estimation with CMU Model =================================================================

    # This code defintely works for hand pose estimation but it is very slow.
    # frameCopy = frame

    # frameWidth = frame.shape[1]
    # frameHeight = frame.shape[0]
    # aspect_ratio = frameWidth/frameHeight
    # inHeight = 368
    # inWidth = int(((aspect_ratio*inHeight)*8)//8)

    # inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
    # net.setInput(inpBlob)
    # output = net.forward()
    # # print(output)
    
    # points = []
    # for i in range(nPoints):
    #     # confidence map of corresponding body's part.
    #     probMap = output[0, i, :, :]
    #     probMap = cv2.resize(probMap, (frameWidth, frameHeight))
    #     # Find global maxima of the probMap.
    #     minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

    #     if prob > threshold :

    #         cv2.circle(frameCopy, (int(point[0]), int(point[1])), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)

    #         cv2.putText(frameCopy, "{}".format(i), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

    #         # Add the point to the list if the probability is greater than the threshold

    #         points.append((int(point[0]), int(point[1])))

    #     else :

    #         points.append(None)
    if not histogram_made:
        frame = draw_rect(frame)
    else:
        if frame_wait > 0:
            frame = cv2.putText(frame, "Saved Skin Tones", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            frame_wait = frame_wait - 1

    if not histogram_made:
        # Display the resulting frame
        cv2.imshow("Live Feed", rescale_frame(frame))
    else:
        # Display the resulting frame
        histMask = hist_masking(frame, hand_histogram)
        bgSubMask = bgSubMasking(frame)
        mask = cv2.bitwise_and(histMask, bgSubMask)
        if not contour_mode:
            cv2.imshow("Live Feed", rescale_frame(mask))
        else:
            thresh = threshold(mask)

            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                maxContour = getMaxContours(contours)
                hull = cv2.convexHull(maxContour)
                mask = cv2.drawContours(mask, [maxContour], 0, (0, 0, 255), 3)
                mask = cv2.drawContours(mask, [hull], 0, (0, 255, 0), 2)
                mask = find_centroid(maxContour, mask)
                ret, num = countFingers(maxContour)
                
                if ret:
                    num = rolling_finger_count(num)
                    if (num < 0):
                        num = 1
                    elif (round(num) != 0):
                        num = round(num) + 1
                    else:
                        num = round(num)
                    phrase = str(num) + "fingers"
                    if num == 1:
                        highest_point = tuple(maxContour[maxContour[:, :, 1].argmin()][0])
                        x1, y1 = highest_point
                        gesture.append((x1, y1))
                    mask = cv2.putText(mask, phrase, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    if len(gesture) > 0:
                        prevPoint = None
                        currPoint = None
                        for i in range(len(gesture)):
                            if (i == 0):
                                prevPoint = gesture[i]
                            else:
                                currPoint = gesture[i]
                                x1, y1 = prevPoint
                                x2, y2 = currPoint
                                mask = cv2.line(mask, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
                                prevPoint = currPoint
                    
            cv2.imshow("Live Feed", rescale_frame(mask))
            
    
    k = cv2.waitKey(1) & 0xFF

    if k == ord('q'):

        break

    if k == ord('g'):
        gesture = []
        


    if k == ord('c'):
        contour_mode = True

    if k == ord('s'):
        bgSubtractor = cv2.createBackgroundSubtractorMOG2(history=10, varThreshold=30, detectShadows=False)

    if k == ord('a'):
        if histogram_made == False:
            print("pressed a")
            histogram_made = True
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            hand_histogram = hand_histogram(frame)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

