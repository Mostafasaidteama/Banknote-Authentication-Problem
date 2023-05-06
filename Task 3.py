import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC as svc
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score ,precision_score ,recall_score
from sklearn.model_selection import cross_val_score
from sklearn import metrics

colnames = ['Variance ','skewness','curtosis','emtropy','class']
Dataset = pd.read_csv('data_banknote_authentication.txt')

features = Dataset.iloc[:, 0:4].values 
lablels = Dataset.iloc[:, 4].values

#Training and Test datasets
train_features, test_features, train_labels, test_labels = train_test_split(features, lablels, test_size=0.2, random_state=21)


#data normalization
normalization = StandardScaler()
train_features=normalization.fit_transform(train_features)
test_features=normalization.transform(test_features)
features=normalization.fit_transform(features)


#SVM with hard margins  
classifier1 = svc(kernel='linear',C=1) 
classifier1.fit(train_features, train_labels)
predicted_labels = classifier1.predict(test_features)


confusion_matrix1= confusion_matrix( test_labels,  predicted_labels)
TP = confusion_matrix1[1, 1]
TN = confusion_matrix1[0, 0]
FP = confusion_matrix1[0, 1]
FN = confusion_matrix1[1, 0]
TNR = TN/(TN+FP)
FPR = FP/(FP+TN)
 
#calculate percentage of accuracy
Accuracy=accuracy_score(test_labels, predicted_labels)
precision =precision_score(test_labels,predicted_labels)
recall=recall_score(test_labels, predicted_labels)
acc_1 =cross_val_score(estimator=classifier1,X=features,y=lablels,cv=5)


print("confusion matrix ="," \n"," \n",confusion_matrix1, " \n") 
print("the Accuracy : ",Accuracy*100,"%"," \n")
print("the precision : ",precision*100,"%"," \n")
print("the sensitivity :",recall*100,"%"," \n")
print("FPR : ",FPR*100,"%"," \n")
print("specifity =",TNR*100,"%"," \n")
print("Accuracy for cross validation : {:.2f} %".format(acc_1.mean()*100)," \n")
print("Standard Deviation: {:.2f} %".format(acc_1.std()*100),"\n","\n")



#SVM soft margins

classifier2 = svc(kernel='linear',C=0.002) 
classifier2.fit(train_features, train_labels)
predicted_labels2 = classifier2.predict(test_features)


confusion_mat2= confusion_matrix( test_labels,  predicted_labels2)
TP2 = confusion_mat2[1, 1]
TN2 = confusion_mat2[0, 0]
FP2 = confusion_mat2[0, 1]
FN2 = confusion_mat2[1, 0]
TNR2 = TN2/(TN2+FP2)
FPR2 = FP2/(FP2+TN2)

 
#calculate percentage of accuracy
Accuracy2=accuracy_score(test_labels, predicted_labels)
precision2=precision_score(test_labels,predicted_labels)
recall2=recall_score(test_labels, predicted_labels)
acc_2 =cross_val_score(estimator=classifier2,X=features,y=lablels,cv=5)


print("Accuracy for cross validation: {:.2f} %".format(acc_2.mean()*100)," \n")
print("Standard Deviation: {:.2f} %".format(acc_2.std()*100)," \n")
print("confusion matrix ="," \n"," \n",confusion_mat2, " \n") 
print("the Accuracy :",Accuracy2*100,"%"," \n")
print("the precision :",precision2*100,"%"," \n")
print("the sensitivity :",recall2*100,"%"," \n")
print("FPR : ",FPR2*100,"%"," \n")
print("specifity :",TNR2*100,"%")


metrics.plot_roc_curve(classifier1,test_features, test_labels)
metrics.plot_roc_curve(classifier2,test_features, test_labels)
