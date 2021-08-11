'''
    File name: imageQuality.py
    Author: maayan wislizky
    Date created: 3/8/2021
    Date last modified: 3/8/2021
    Python Version: 3.7
'''


import shutil

import imquality.brisque as brisque
import PIL.Image

import glob
import  os

import numpy as np

image_list_score = []
for filename in glob.glob(r'C:\Users\maayan\Pictures\cycleGanProj\selectedimages/*.jpg'):
    img = PIL.Image.open(filename)
    score = brisque.score(img)
    print(score, filename)
    image_list_score.append([float(score), filename])

cc = sorted(image_list_score, key=lambda x : x[0])
ccc = np.array(cc)[:,1][:100]
for i in ccc:
    shutil.move(i, r'C:\Users\maayan\Pictures\cycleGanProj\test\testA/')


