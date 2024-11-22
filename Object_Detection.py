# importing lib
import cv2
import numpy as np

#To load pre-trained models
net = cv2.dnn.readNet(r'yolov3.weights', r'yolov3.cfg')


classes = [] #this will store the list of class names in file coco.names


        #file handling (NOTE:- f here is used  )
with open(r'coco.names',"r") as f:
         #f.read() is used to read the given file and split all the lines. 
    classes = f.read().splitlines()
#Other way but its manual
# f = open(r"F:\College projects\Object Detection\coco.names", "r")
# try:
#     classes = f.read().splitlines()
# finally:
#     f.close()

        #To capture video from the webcam 
capture = cv2.VideoCapture(0) # 0 is the default camera

        #font is used to display text on images
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

        #color is used to generate random color around objects
color = np.random.uniform(0, 255, size=(100,3))

        #created an endless loop to run camera
while True:
    buul, img = capture.read()#buul here is boolean value which tells us if the image is found or not 
    if not buul:
        break
    
    height, width, _ = img.shape# these are the h,w of image and _ is the placeholder for color channels 
        #blob from image is used to
    blob = cv2.dnn.blobFromImage(img, 1/255, (416,416), (0,0,0), swapRB=True, crop=False)
    
    net.setInput(blob)

    output = net.getUnconnectedOutLayersNames()
    layeroutput = net.forward(output)
        
        #created empty lists to store values
    boxes = []          # to store values of detection boxes
    confidence = []     # to store values of confidence
    class_ids = []      # store maximum index of the prediction

    for output in layeroutput:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidences = scores[class_id]

            if confidences > 0.2:
                cen_x = int(detection[0]*width)
                cen_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(cen_x - w/2)
                y = int(cen_y - h/2)

                boxes.append([x,y,w,h])
                confidence.append(float(confidences))
                class_ids.append(class_id)



    indexes = cv2.dnn.NMSBoxes(boxes, confidence, 0.2 ,0.4)   

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidences = str(round(confidence[i], 2))
            colors = color[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), colors, 2)
            cv2.putText(img, label + " " + confidences, (x, y + 20), font, 2, (255, 255, 255), 2)

    cv2.imshow('Image', img)
    key = cv2.waitKey(1)
    if key == 27:
        break

capture.release()
cv2.destroyAllWindows()





