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
        self.float_index=np.where(x.dtypes!='object')[0]
        self.object_index=np.where(x.dtypes=='object')[0]
        num=Y.value_counts()
        self.prior = (num / np.sum(num)).to_dict() #{' <=50K': 0.7591904425539756, ' >50K': 0.2408095574460244}
    def predict(self, x):
            P = []
            for Y_form, prior in (self.prior).items():
                Y_index = (self.Y == Y_form).values
                p = prior
                i = 0
                while i < len(self.X.columns):
                    if  i in self.float_index:
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
                    i=i.strip()
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