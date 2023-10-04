from scipy.spatial import distance
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from preprocess import Data
df3=pd.read_csv('adult_names.txt',sep=':',header=None)
a=Data('adult_train.csv','adult_test.csv',df3[0],labelColumn='label',OrdinalColumns={' <=50K':0,' >50K':1})
data_train,data_test,y_train,y_test=a.encoder()
print(data_train,data_test)
data1,data2=train_test_split(data_train)
def EuclideanDistances(A, B):
    BT = B.transpose()
    vecProd = np.dot(A,BT)
    SqA =  A**2
    sumSqA = np.matrix(np.sum(SqA, axis=1))
    sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1]))
    SqB = B**2
    sumSqB = np.sum(SqB, axis=1)
    sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))    
    SqED = sumSqBEx + sumSqAEx - 2*vecProd
    SqED[SqED<0]=0.0   
    ED = np.sqrt(SqED)
    return ED
d=EuclideanDistances(data1,data2)
dist=np.max(d,axis=1)
far=np.array(sorted(dist)).reshape(24420,1)[int(0.9*len(dist))]
near=np.array(sorted(dist)).reshape(24420,1)[int(0.1*len(dist))]
g=np.sqrt((far**2-near**2)/(2*(np.log(far)**2-np.log(near)**2)))
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
tuned_parameters={'C': list(range(1,3)),'gamma':list(range(1,3))}
score= 'precision'
clf = GridSearchCV(SVC(kernel='rbf'), tuned_parameters, cv=5,
                    scoring='%s_macro' % score)
clf.fit(data_train,y_train)
mean_score=clf.cv_results_['mean_test_score']
tuned_parameters={'C': list(range(1,3))}
score= 'precision'
clf = GridSearchCV(SVC(kernel='rbf',gamma=1/g[0]), tuned_parameters, cv=5,
                    scoring='%s_macro' % score)
clf.fit(data_train,y_train)
mean_score=clf.cv_results_['mean_test_score']
tuned_parameters={'C': list(range(1,3))}
score= 'precision'
clf = GridSearchCV(SVC(kernel='linear',gamma=1/g[0]), tuned_parameters, cv=5,
                    scoring='%s_macro' % score)
clf.fit(data_train,y_train)
mean_score=clf.cv_results_['mean_test_score']
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.metrics import precision_score
gnb = GaussianNB()
gnb.fit(data_train,y_train)
precision_score(y_test,gnb.predict(data_test))