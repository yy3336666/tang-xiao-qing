from scipy.spatial import distance
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from p import Data
df3=pd.read_csv('adult_names.txt',sep=':',header=None)
a=Data('adult_train.csv','adult_test.csv',df3[0],labelColumn='label',OrdinalColumns={' <=50K':0,' >50K':1})
df1,df2=a.readcsv()
data_train,y_train=a.encoder(df1)
data_test,y_test=a.encoder(df2)
data1,data2=train_test_split(data_train)
d=distance.cdist(data1,data2)
dist=np.max(d,axis=1)
print(dist)
far=np.array(sorted(dist)).reshape(24420,1)[int(0.9*len(dist))]
near=np.array(sorted(dist)).reshape(24420,1)[int(0.1*len(dist))]
g=np.sqrt((far**2-near**2)/(2*(np.log(far)**2-np.log(near)**2)))
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score
tuned_parameters={'C': list(range(1,3)),'gamma':list(range(1,3))}
score= 'precision'
clf = GridSearchCV(SVC(kernel='rbf'), tuned_parameters, cv=5,
                    scoring='%s_macro' % score)
clf.fit(data_train,y_train)
mean_score=clf.cv_results_['mean_test_score']
print(mean_score)
tuned_parameters={'C': list(range(1,3))}
clf = GridSearchCV(SVC(kernel='rbf',gamma=1/g[0]), tuned_parameters, cv=5,
                    scoring='%s_macro' % score)
clf.fit(data_train,y_train)
mean_score=clf.cv_results_['mean_test_score']
print(mean_score)
tuned_parameters={'C': list(range(1,3))}
clf = GridSearchCV(SVC(kernel='linear',gamma=1/g[0]), tuned_parameters, cv=5,
                    scoring='%s_macro' % score)
clf.fit(data_train,y_train)
mean_score=clf.cv_results_['mean_test_score']
print(mean_score)
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.metrics import precision_score
gnb = GaussianNB()
gnb.fit(data_train,y_train)
print(precision_score(y_test,gnb.predict(data_test)))