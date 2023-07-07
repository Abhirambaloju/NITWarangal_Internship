import numpy as np
import pandas as pd
import cv2
from PIL import Image
#import scipy
import itertools
# import tensorflow as tf
# from tensorflow.keras.applications import *
# from tensorflow.keras.optimizers import *
# from tensorflow.keras.losses import *
# from tensorflow.keras.layers import *
# from tensorflow.keras.models import *
# from tensorflow.keras.callbacks import *
# from tensorflow.keras.preprocessing.image import *
# from tensorflow.keras.utils import *
# # import pydot
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import *
# from sklearn.model_selection import *
# import tensorflow.keras.backend as K
# from sklearn import metrics
# from tqdm import tqdm, tqdm_notebook
# from colorama import Fore
# import json
import matplotlib.pyplot as plt
#import seaborn as sns
from glob import glob
# from skimage.io import *
# #%config Completer.use_jedi = False
# import time
# from sklearn.decomposition import PCA
# from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
# import lightgbm as lgb
# from xgboost import XGBClassifier
# from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier
# from sklearn.pipeline import make_pipeline
# from sklearn.pipeline import Pipeline
# from sklearn.neighbors import *
# print("All modules have been imported")

# def plot_confusion_matrix(cm, classes,
#                           normalize=False,
#                           title='Confusion matrix',
#                           cmap=plt.cm.YlOrRd):
#     plt.figure(figsize = (6,6))
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=90)
#     plt.yticks(tick_marks, classes)
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

#     thresh = cm.max() / 2.
#     cm = np.round(cm,2)
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, cm[i, j],
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.show()
    
    
info=pd.read_csv("C:/Users/Abhiram/Downloads/info.txt",sep=" ")
info=info.drop('Unnamed: 7',axis=1)
info.SEVERITY.fillna(0)

from PIL import Image
import glob
x= []
x1=[]
clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
for filename in sorted(glob.glob("C:/Users/Abhiram/Downloads/all-mias/*.pgm")): 
    img=cv2.imread(filename)
    img =cv2.resize(img,(224, 224))
    x.append(img)
    
for filename in sorted(glob.glob("C:/Users/Abhiram/Downloads/all-mias/*.pgm")): 
      img=cv2.imread(filename)
      img =cv2.resize(img,(224, 224))
      x.append(img)
      median = cv2.medianBlur(img,1)
      x1.append(median)
      # clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(224,224))
      # cl1 = clahe.apply(img)
      # x1.append(cl1)
   
    
    

fig=plt.figure(figsize=(15,20))
columns = 4
rows = 4
for i in range(1, columns*rows +1):
    img = np.random.randint(20)
    fig.add_subplot(rows, columns, i)
    plt.imshow(x1[i])
plt.show()