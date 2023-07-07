import cv2
import glob
from matplotlib import pyplot as plt
import numpy as np 
import os
import pandas as pd

inputFolder='C:/Users/Abhiram/.spyder-py3/images'
os.mkdir('Resized Folder')

i=0

for img in glob.glob(inputFolder + "/*.jpg"):
    image=cv2.imread(img)
    imgResized=cv2.resize(image,(400,400))
    cv2.imwrite("Resized Folder/ image%i.jpg"%i,imgResized)
    i+=1
    cv2.imshow('image',imgResized)
    cv2.waitKey(0)
cv2.destroyAllWindows()

