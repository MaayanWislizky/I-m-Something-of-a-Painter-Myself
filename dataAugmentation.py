'''
    File name: dataAugmentation.py
    Author: maayan wislizky
    Date created: 1/8/2021
    Date last modified: 1/8/2021
    Python Version: 3.7
'''

import glob

from PIL import Image
from random import randrange
import  os
from random import randint

# Add padding
matrix = 100

sample_list1 = []
sample_list = []
srcPath = r'C:\Users\maayan\Pictures\cycleGanProj\selectedMonet/'
dstPath = r'C:\Users\maayan\Pictures\cycleGanProj\aug/'
for filename in glob.glob(srcPath+'*.jpg'):
    sample_list.append(filename)

for i in sample_list:
    img = Image.open(i)
    x, y = img.size
    x1 = randrange(0, x - matrix)
    y1 = randrange(0, y - matrix)
    fn  =  os.path.split(i)
    sample_list1.append([fn[1], img.crop((x1, y1, x1 + matrix, y1 + matrix))])


for i in sample_list1:
    i[1].save(dstPath+ str(randint(1,1000))+"_"+i[0] )




