import pandas as pd
from sklearn.preprocessing import LabelEncoder
class Data():
    def __init__(self,trainPath=None,testPath=None,columns=None,labelColumn=None,OrdinalColumns={}):
        self.trainpath=trainPath
        self.testPath=testPath
        self.columns=columns
        self.labelColumn=labelColumn
        self.OrdinalColumns=OrdinalColumns 
    def readcsv(self):
        df1=pd.read_csv(self.trainpath,header=None)
        df2=pd.read_csv(self.testPath,header=None)
        df1.rename(columns=self.columns,inplace=True)
        df2.rename(columns=self.columns,inplace=True)
        return df1,df2
    def encoder(self):
        df1,df2=self.readcsv()
        col=df1.dtypes[df1.dtypes=='object'].index
        df1[self.labelColumn]=df1[self.labelColumn].map(self.OrdinalColumns)
        df2[self.labelColumn]=df2[self.labelColumn].map(self.OrdinalColumns)
        col=col.drop('label')
        col_label=df1[col].nunique()[df1[col].nunique()==2].index
        col_one=df1[col].nunique()[df1[col].nunique()!=2].index
        le=LabelEncoder()
        df11=df1.copy()
        df22=df1.copy()
        for i in col_label:
            df11[i]=le.fit_transform(df11[i])
            df22[i]=le.fit_transform(df22[i])
        data_train=df11.join(pd.get_dummies(df11[col_one]))
        data_test=df22.join(pd.get_dummies(df22[col_one]))
        data_train.drop(columns=col_one,inplace=True)
        data_test.drop(columns=col_one,inplace=True)
        y_train=df1[self.labelColumn]
        y_test=df2[self.labelColumn]
        data_test.drop(columns=self.labelColumn,inplace=True)
        data_train.drop(columns=self.labelColumn,inplace=True)
        return data_train,data_test,y_train,y_test
if __name__=='__main__':
    df3=pd.read_csv('adult_names.txt',sep=':',header=None)
    a=Data('adult_train.csv','adult_test.csv',df3[0],labelColumn='label',OrdinalColumns={' <=50K':0,' >50K':1})
    data_train,data_test,y_train,y_test=a.encoder()
    print(data_train,data_test)
