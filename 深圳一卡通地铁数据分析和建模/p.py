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
    def encoder(self,df):
        col=df.dtypes[df.dtypes=='object'].index
        df[self.labelColumn]=df[self.labelColumn].map(self.OrdinalColumns)
        col=col.drop('label')
        col_label=df[col].nunique()[df[col].nunique()==2].index
        col_one=df[col].nunique()[df[col].nunique()!=2].index
        le=LabelEncoder()
        df11=df.copy()
        for i in col_label:
            df11[i]=le.fit_transform(df11[i])
        data=df11.join(pd.get_dummies(df11[col_one]))
        data.drop(columns=col_one,inplace=True)
        y=df[self.labelColumn]
        data.drop(columns=self.labelColumn,inplace=True)
        return data,y
if __name__=='__main__':
    df3=pd.read_csv('adult_names.txt',sep=':',header=None)
    a=Data('adult_train.csv','adult_test.csv',df3[0],labelColumn='label',OrdinalColumns={' <=50K':0,' >50K':1})
    df1,df2=a.readcsv()
    x_train,y_train=a.encoder(df1)
    x_test,y_test=a.encoder(df2)
    print(x_train,x_test)

