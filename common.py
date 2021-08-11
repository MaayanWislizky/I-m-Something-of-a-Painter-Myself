'''
    File name: common.py
    Author: maayan wislizky
    Date created: 8/8/2021
    Date last modified: 8/8/2021
    Python Version: 3.7
'''


import  random
from PIL import Image
import glob
import os
import shutil


def renameFileName(src,dst):
    for count, filename in enumerate(os.listdir(src)):
        dst =dst+"/" + str(count) + ".jpg"
        src =src+"/"+ filename

        # rename() function will
        # rename all the files
        os.rename(src, dst)

def moveImgFiles(srcDir,destDir, amount=200):
    images = glob.glob(srcDir+'/*.jpg')
    for i in range(amount):
        random_image = random.choice(images)
        fn  =  os.path.split(random_image)
        shutil.move(random_image, destDir+'/'+fn[1])
       # // os.replace(random_image, r'C:\Users\maayan\Pictures\cycleGanProj\test\testA/'+fn[1])


def resizeNsave(src,dst,size=256):
    image_list = []
    for filename in glob.glob(src+'/*.jpg'):
        img=Image.open(filename)
        img = img.resize((size, size), Image.ANTIALIAS)
        fn  =  os.path.split(filename)
        name = fn[1]+"_crp"
        filenamenew =dst+"/aug_"+name+".jpg"
        img.save(filenamenew)



