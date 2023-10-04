import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
class NaiveBayes:
    def __init__(self):
        self.variable_type = dict()
        self.prior = dict()
        self.X = 0
        self.Y = 0

    def fit(self, X, Y):
        self.X = X
        self.Y = Y

        columns = X.columns
        for item in columns:
            if X[item].dtypes in ['float64' ,'float32']:
                self.variable_type[item] = 'float'
            else:
                self.variable_type[item] = 'object'

        Y = Y.to_frame() #将系列对象转换为DataFrame
        num = Y.groupby(list(Y.columns))[list(Y.columns)[0]].agg('count')
        self.prior = (num / np.sum(num)).to_dict() #{' <=50K': 0.7591904425539756, ' >50K': 0.2408095574460244}

    def predict(self, x, **argv):
        if isinstance(x, pd.Series): #判断一个对象是否是一个已知的类型
            P = dict()
            for Y_form, prior in (self.prior).items():
                Y_index = (self.Y == Y_form).values
                p = prior
                for i, value in enumerate(x): #同时列出数据和数据下标
                    form = list(self.variable_type.values())[i]
                    if form == 'float':
                        temp_mean = np.mean(self.X.iloc[Y_index, i])
                        temp_std = np.std(self.X.iloc[Y_index, i])
                        p = p * (np.exp(-(value - temp_mean) ** 2 / (2 * temp_std ** 2)) / (np.sqrt(2 * np.pi) * temp_std))
                    else:
                        temp_dict = (self.X.iloc[Y_index, :].groupby([self.X.columns[i]])[self.X.columns[i]].agg(
                            'count')).to_dict()
                        p = p * (temp_dict.get(value, 1e-4) / np.sum(list(temp_dict.values())))

                P[Y_form] = p
            for item in P.keys():
                P[item] = P[item] / np.sum(list(P.values()))

            return P

        else:
            P = []
            for Y_form, prior in (self.prior).items():
                Y_index = (self.Y == Y_form).values
                p = prior
                i = 0
                while i < len(self.X.columns):
                    form = list(self.variable_type.values())[i]
                    if form == 'float':
                        tmp_mean = np.mean(self.X.iloc[Y_index, i])
                        tmp_std =  np.std(self.X.iloc[Y_index, i])
                        p = p * (np.exp(-(np.array(x.iloc[:, i]) - tmp_mean) ** 2 / (2 * tmp_std ** 2)) / (
                                np.sqrt(2 * np.pi) * tmp_std))
                        i += 1
                    else:
                        tmp_dict = (self.X.iloc[Y_index, :].groupby([self.X.columns[i]])[self.X.columns[i]].agg(
                            'count')).to_dict()
                        p = p * (np.array(list(map(lambda x: tmp_dict.get(x, 1e-4), np.array(x.iloc[:, i])))) / np.sum(
                            list(tmp_dict.values())))
                        i += 1
                P.append(p)
            return pd.DataFrame(data=(P / np.sum(P, axis=0)).T, columns=(self.prior).keys())

    def score(self, tX, tY, criteria="accuracy"):
        Y_temp = self.predict(tX)
        if isinstance(Y_temp, dict):
            if criteria == "accuracy":
                Y = max(Y_temp, key=Y_temp.get)
                tY=tY.str.strip()
                tY=tY.str.replace('.','')
                y1=[]
                for i in Y:
                        y1.append(i.replace(" ",""))
                print(tY.unique())
                return accuracy_score([tY], [y1])
            else:
                Y = Y_temp[max(Y_temp, key=Y_temp.get)]
                y2=[]
                tY=tY.str.strip()
                tY=tY.str.replace('.','')
                for i in Y:
                        y2.append(i.replace(" ",""))
                return roc_auc_score([tY], [y2])

        else:
            if criteria == "accuracy":
                Y = np.argmax(np.array(Y_temp), axis=1)
                Y = np.array(Y_temp.columns)[Y]
                y3=[]
                tY=tY.str.strip()
                tY=tY.str.replace('.','')
                for i in Y:
                        i=i.strip()
                        y3.append(i.replace(".",""))
                return accuracy_score(tY, y3)
            else:
                tY = list(map(lambda x: np.argwhere(np.array(Y_temp.columns) == x)[0,0], tY.values))
                Y = Y_temp.loc[:, Y_temp.columns[1]]
                tY=tY.str.strip()
                tY=tY.str.replace('.','')
                y4=[]
                for i in Y:
                        y4.append(i.replace(".",""))
                return roc_auc_score(tY, y4)


if __name__ == '__main__':
    data1 = pd.read_csv('adult_train.csv', header=None)
    Y = data1.iloc[:, -1]
    X = data1.iloc[:, :-1]
    X.iloc[:, [0, 2, 4, 10, 11, 12]] = X.iloc[:, [0, 2, 4, 10, 11, 12]].astype('float64')

    data2 = pd.read_csv('adult_test.csv', header=None)
    y = data2.iloc[:, -1]  #将adult_test.csv最后一列的.删除
    x = data2.iloc[:, :-1]
    x.iloc[:, [0, 2, 4, 10, 11, 12]] = x.iloc[:, [0, 2, 4, 10, 11, 12]].astype('float64')

    answer = NaiveBayes()
    answer.fit(X,Y)
    print(answer.predict(x))
    print('计算准确率为:', answer.score(x,y,'accuracy'))
    
