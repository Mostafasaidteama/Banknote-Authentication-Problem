import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from random import seed
from random import random
from math import exp


from matplotlib.pyplot import rcParams

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler , MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC as svc
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score ,precision_score ,recall_score
from sklearn.model_selection import cross_val_score
from sklearn import metrics

from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras import layers
from tensorflow.keras import initializers


Dataset = pd.read_csv('data_banknote_authentication.csv')

features = Dataset.drop(['class'],axis = 1)
lablels = Dataset['class']

#Training and Test datasets
Xtrain,Xtemp,Ytrain,Ytemp = train_test_split(features,lablels, test_size=0.3,random_state=42,shuffle=True)
Xval,Xtest,Yval,Ytest = train_test_split(Xtemp,Ytemp, test_size=0.5,random_state=42)


#data normalization
normalization = StandardScaler()
Xtrain_scaled = normalization.fit_transform(Xtrain)
Xtrain_scaled = pd.DataFrame(Xtrain_scaled,columns= features.columns)

Xtest_scaled = normalization.fit_transform(Xtest)
Xtest_scaled = pd.DataFrame(Xtest_scaled,columns= features.columns)

Xval_scaled = normalization.fit_transform(Xval)
Xval_scaled = pd.DataFrame(Xval_scaled,columns= features.columns)


#Defining Model Layers
Model = Sequential()
Model.add(Dense(256,input_shape=(4,), kernel_initializer = initializers.RandomNormal(), bias_initializer= initializers.Zeros(),activation = 'relu'))
Model.add(Dense(64, activation = 'relu'))
Model.add(Dense(64, activation = 'sigmoid'))


#Compiling Model and Testing
Model.compile(loss='mean_squared_error',optimizer = 'sgd')
setpsPerEpoch = 50

history = Model.fit(Xtrain_scaled,Ytrain,batch_size=10,epochs=50,steps_per_epoch = setpsPerEpoch,validation_data=(Xval_scaled,Yval),)

Model.evaluate(Xtest,Ytest)


#Plotting MSE
loss_train = history.history['loss']
epochs = range(0,50)
plt.plot(epochs,loss_train,'g',label = 'Training Loss')

plt.title('MSE loss')
plt.xlabel('Epoch')
plt.ylabel('Error (%)')
plt.legend()
plt.show()


#SVM with hard margins  
classifier1 = svc(kernel='linear',C=1) 
classifier1.fit(Xtrain_scaled, Ytrain)
predicted_labels = classifier1.predict(Xtest_scaled)


confusion_matrix1= confusion_matrix( Ytest,  predicted_labels)
TP = confusion_matrix1[1, 1]
TN = confusion_matrix1[0, 0]
FP = confusion_matrix1[0, 1]
FN = confusion_matrix1[1, 0]
TNR = TN/(TN+FP)
FPR = FP/(FP+TN)
 

#calculate percentage of accuracy
Accuracy=accuracy_score(Ytest, predicted_labels)
precision =precision_score(Ytest,predicted_labels)
recall=recall_score(Ytest, predicted_labels)
acc_1 =cross_val_score(estimator=classifier1,X=features,y=lablels,cv=5)


#Compiling Model and Testing
Model.compile(loss='mean_squared_error',optimizer = 'sgd')
setpsPerEpoch = 50

history = Model.fit(Xtest_scaled,Ytest,batch_size=10,epochs=50,steps_per_epoch = setpsPerEpoch,validation_data=(Xval_scaled,Yval),)

Model.evaluate(Xtest,Ytest)


#Plotting proportion of misclassifications
loss_train = history.history['loss']
epochs = range(0,22)
plt.plot(epochs,loss_train,'g',label = 'Training Loss')

plt.title('Network error')
plt.xlabel('Epoch')
plt.ylabel('Error (%)')
plt.legend()
plt.show()


#Plotting ROC curve
metrics.plot_roc_curve(classifier1,Xtrain_scaled, Ytrain)