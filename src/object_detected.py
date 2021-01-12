import cv2
import numpy as np
import keras
import os






# Load YOLO  for objects localization 
net = cv2.dnn.readNet("../yolo/yolov3.weights", "../yolo/yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]



# Create a dictionary  with classes and indices
train_path = "../data/train/"
classes = {}
indice = 0
for folder in os.listdir(train_path):
    classes[folder] = indice
    indice += 1
# create function to get classe
def get_classe(n):
    for classe , value in classes.items():
        if n == value:
            return classe


test_cnn_model_2 = keras.models.load_model("../model/model_v2.model")



# Load image for test
img = cv2.imread("../data/predict/room_desktop.png")
img = cv2.resize(img, (800,600))
height, width, chanel = img.shape
print(height, width, chanel)
print (img)

# Detecting Object
blob = cv2.dnn.blobFromImage(img, 0.00392, size = (416,416),  swapRB = True, crop = False )


# Use image as Input in the blob format
net.setInput(blob)
net_output = net.forward(output_layers)
# print(len(net_output[0][400]))


bounding_boxes =[]
confidences = []

for detection in net_output:
    for obj in detection:
        # scores = obj[5:]
        # class_id = np.argmax(scores)
        confidence = obj[4]
        if confidence > 0.979:
            # Object detected
            x_center = int(obj[0] * width)
            y_center = int(obj[1] * height)
            w = int(obj[2] * width)
            h = int(obj[3] * height)
            # cv2.circle(img, (x_center, y_center), 5, (0.0,255),2)

            # Rectangle coordinates for each object detected
            x = int(x_center - w / 2)
            y = int(y_center - h / 2 )

            bounding_boxes.append( [x, y, w, h])
            confidences.append(float(confidence))
            
            
            
indexes = cv2.dnn.NMSBoxes(bounding_boxes, confidences, 0.5, 0.4)  
print(indexes)    
number_of_obj_detected = len(bounding_boxes)
font = cv2.FONT_HERSHEY_PLAIN

for box in range(number_of_obj_detected):
    x, y, w, h = bounding_boxes[box]
    if box in indexes:
        
        img_roi = img[y:y+h, x:x+w]
        img_roi_resized = cv2.resize(img_roi, (32,32))
        img_roi_normalized=img_roi_resized / 255.0
        img_roi_reshaped = np.reshape(img_roi_normalized,(1,32,32,3))
        img_pred = test_cnn_model_2.predict(img_roi_reshaped)
       
        label = get_classe(np.argmax(img_pred))
        print(label)
        
        # cv2.imshow("Image", img_roi_normalized)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
      
        
       
        cv2.rectangle(img, (x, y), (x + w, y + h), (0,0,255), 2)
        cv2.putText(img, label, (x, y+20), font, 1 , (0, 0, 0) ,2)










cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
