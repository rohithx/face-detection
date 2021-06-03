# Written by Rohith
import os
import cv2
import json
import sys

impath = sys.argv[1]

# Setting up the classifier
cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
fin_list = []

# Finding the bounding box for each image in the given folder
for file in os.listdir(impath):
    img = cv2.imread(os.path.join(impath,file))
    bwimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(bwimg, 1.2, 4)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)
        res = {"iname": file, "bbox": [int(x), int(y), int(w), int(h)]}
        fin_list.append(res)

# Dumping the JSON file
with open("results.json",'w') as fl:
    json.dump(fin_list, fl) 