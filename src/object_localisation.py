import cv2
import numpy as np
import os

class Localization():
    """
    docstring
    """
    
    
    def __init__(self):
        self.classes = {}
        self.net = None
        self.output_layers = None
        self.image = None
        self.height = 0
        self.width = 0
        self.chanel = 0
        self.net_output = None
        self.bounding_boxes = []
        self.confidences = []
        self.indexes = None
        self.number_of_obj_detected = 0



    # def create_dict(self, path_train):
    #     """
    #     Create dictionary with names of classes and indices
    #     """
    #     indice = 0
    #     for folder in os.listdir(path_train):
    #         self.classes[folder] = indice
    #         indice += 1
    #     return self.classes
    
    def get_classe(self, n):
        """
        return classe of image
        """
        self.classes = {'apple':0, 'forest': 1, 'crocodile':2}
        for classe , value in self.classes.items():
            if n == value:
                return classe
    
    def yolo_network(self, weights_path, conf_path):
        """

        """
        self.net = cv2.dnn.readNet(weights_path, conf_path)
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        return self.net, self.output_layers

    def read_image(self, path_img):
        """
        """
        img = cv2.imread(path_img)
        self.image = cv2.resize(img, (800,600))
        self.height, self.width, self.chanel = self.image.shape
        return self.image, self.height, self.width, self.chanel
      

    def get_net_output(self):
        """
        """
        blob = cv2.dnn.blobFromImage(self.image, 0.00392, size = (416,416),  swapRB = True, crop = False )
        # Use image as Input in the blob format
        self.net.setInput(blob)
        self.net_output = self.net.forward(self.output_layers)
        return self.net_output
    

    def get_bounding_box(self):
        """
        docstring
        """
        for detection in self.net_output:
            for obj in detection:
                # scores = obj[5:]
                # class_id = np.argmax(scores)
                confidence = obj[4]
                if confidence > 0.7:
                    # Object detected
                    x_center = int(obj[0] * self.width)
                    y_center = int(obj[1] * self.height)
                    w = int(obj[2] * self.width)
                    h = int(obj[3] * self.height)
                    # cv2.circle(img, (x_center, y_center), 5, (0.0,255),2)

                    # Rectangle coordinates for each object detected
                    x = int(x_center - w / 2)
                    y = int(y_center - h / 2 )

                    self.bounding_boxes.append( [x, y, w, h])
                    self.confidences.append(float(confidence))

        return self.bounding_boxes, self.confidences

    def no_max_boxes(self):
        """
        """
        self.indexes = cv2.dnn.NMSBoxes(self.bounding_boxes, self.confidences, 0.5, 0.4)  
        self.number_of_obj_detected = len(self.bounding_boxes)
        return self.indexes, self.number_of_obj_detected
    
    def object_detection(self, model, image_test_path, weights_path, conf_path):
        """
        """
        self.yolo_network(weights_path,conf_path)
        # self.create_dict(train_path)
        self.read_image(image_test_path)
       
        self.get_net_output()
        self.get_bounding_box()
        self.no_max_boxes()
        
        for box in range(self.number_of_obj_detected):
            x, y, w, h = self.bounding_boxes[box]
            if box in self.indexes:
                
                img_roi = self.image[y:y+h, x:x+w]
                img_roi_resized = cv2.resize(img_roi, (32,32))
                img_roi_normalized = img_roi_resized / 255.0
                img_roi_reshaped = np.reshape(img_roi_normalized,(1,32,32,3))
                img_pred = model.predict(img_roi_reshaped)
            
                label = self.get_classe(np.argmax(img_pred))
            
                
                # cv2.imshow("Image", img_roi_normalized)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
            
                
            
                cv2.rectangle(self.image, (x, y), (x + w, y + h), (0,0,255), 2)
                cv2.putText(self.image, label, (x, y+20), cv2.FONT_HERSHEY_PLAIN, 1 , (0,255,128) ,2)

        cv2.imshow("Image", self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()











    

    

    

        


    
    




    