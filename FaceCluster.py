import face_recognition
import json
import cv2
import os
import random
import numpy as np
import sys

impath = sys.argv[1]
cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
fin_list= []
for file in os.listdir(impath):
    img = cv2.imread(os.path.join(impath,file))
    bwimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(bwimg, 1.2, 4)
    for (x,y,w,h) in faces:
        enc = face_recognition.face_encodings(img,[(y,x+w,y+h,x)])
        res = {"iname": file, "enc": enc}
        fin_list.append(res)

# Identifying k value from folder name
k = int(impath[12:len(impath)])

# Using a random seed
random.seed(11119)

# Randomly choosing the initial centroid
init_cents = random.sample(range(0,len(fin_list)),k)
cents = []
for i in init_cents:
    cents.append(fin_list[i]['enc'])

# Function to assign the closest centroid to each point
def calc_ass(ic, fin_list, k_val):
    init_ass = []
    for i in fin_list:
        tmp = []
        for j in ic:
            t = np.linalg.norm(np.array(i['enc']) - np.array(j))
            tmp.append(t)
        init_ass.append(tmp.index(min(tmp)))   
    return init_ass

# Function to calibrate new centroids
def new_cents(cluster, fin_list, k_val):
    new_c = []
    for j in range(0,k_val):
        indices =  np.where(np.array(cluster) == j)[0]
        f = [fin_list[b]['enc'] for b in indices]
        ini = np.zeros((1,128))
        for e in f:
            ini += e
        ini = ini/len(f)
        new_c.append(ini)
    return new_c        

# Iterating until the centroids are not changing
while True:
    c = calc_ass(cents, fin_list, k)
    nc = new_cents(c, fin_list, k)
    nnnc = calc_ass(nc, fin_list, k)
    if c == nnnc:
        break
    else:
        cents = nc

# Identifying the clusters based on the final centroids    
final_list = []
for i in range(0,k):
    ix = [j for j, x in enumerate(nnnc) if x == i]
    clust = []
    for b in ix:
        clust.append(fin_list[b]['iname'])
    dic = {"cluster_no": i, "elements": clust}
    final_list.append(dic)     

# Dumping the JSON 
with open("clusters.json",'w') as fl:
    json.dump(final_list, fl) 