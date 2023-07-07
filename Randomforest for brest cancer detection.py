#import libraries 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
#Load the data 
#from google.colab import files # Use to load data on Google Colab #uploaded = files.upload() # Use to load data on Google Colab 
df = pd.read_csv('data.csv') 
df.head(7)
#Count the number of rows and columns in the data set
df.shape
#Count the empty (NaN, NAN, na) values in each column
df.isna().sum()
#Drop the column with all missing values (na, NAN, NaN)
#NOTE: This drops the column Unnamed
df = df.dropna(axis=1)
#Get the new count of the number of rows and cols
df.shape
#Get a count of the number of 'M' & 'B' cells
df['diagnosis'].value_counts()
#Visualize this count 
sns.countplot(df['diagnosis'],label="Count")
#Look at the data types 
df.dtypes
#Encoding categorical data values (
from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
df.iloc[:,1]= labelencoder_Y.fit_transform(df.iloc[:,1].values)
print(labelencoder_Y.fit_transform(df.iloc[:,1].values))
sns.pairplot(df, hue="diagnosis")
df.head(5)
#Get the correlation of the columns
df.corr()
plt.figure(figsize=(20,20))  
sns.heatmap(df.corr(), annot=True, fmt='.0%')
X = df.iloc[:, 2:31].values 
Y = df.iloc[:, 1].values 
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# def models(X_train,Y_train):
  
#   #Using RandomForestClassifier method of ensemble class to use Random Forest Classification algorithm
#   from sklearn.ensemble import RandomForestClassifier
#   forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
#   forest.fit(X_train, Y_train)
  
#   #print model accuracy on the training data.
  
#   print('[6]Random Forest Classifier Training Accuracy:', forest.score(X_train, Y_train))
  
#   return  forest

# model = models(X_train,Y_train)



from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
# for i in range(len(model)):
#   print('Model ',i)
#   #Check precision, recall, f1-score
#   print( classification_report(Y_test, model[i].predict(X_test)) )
#   #Another way to get the models accuracy on the test data
#   print( accuracy_score(Y_test, model[i].predict(X_test)))
#   print()#Print a new line



from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
forest.fit(X_train, Y_train)

X_train_prediction = forest.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy on training data = ', training_data_accuracy)

x_test_prediction = forest.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, x_test_prediction)
print('Accuracy on test data = ', test_data_accuracy)

