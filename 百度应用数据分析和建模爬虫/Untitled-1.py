# %%
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
import pyecharts.charts as pyec
from pyecharts import options as opts



# %%
path='2020年1月百度迁徙数据/迁出'
path1='2020年1月百度迁徙数据/迁入'
files=os.listdir(path)#获得迁出文件夹下内容
files1=os.listdir(path1)#获得迁入文件夹下内容

# %%
data=[]
data1=[]
des=[]
des1=[]
for file in files:
    df=pd.read_csv(path+'/'+file)
    for i in df.columns[1:]:
        df.loc[:,i].fillna(df.loc[:,i].mean(),inplace=True)#对所有城市进行平均值填充
    df.fillna(0,inplace=True)#对其他没有数据的城市用0来填充
    print(df.describe())#获得每个表格的数据情况
    data.append(df)
    des.append(df.describe())
for file in files1:
    df=pd.read_csv(path1+'/'+file)
    for i in df.columns[1:]:
        df.loc[:,i].fillna(df.loc[:,i].mean(),inplace=True)
    df.fillna(0,inplace=True)
    print(df.describe())
    data1.append(df)
    des1.append(df.describe())


# %%
plt.title('迁出30天北京的百度迁徙平均值')
means_beijing=[]
for i in range(len(des)):
    mean=des[i].loc['mean',:][0]
    means_beijing.append(mean)#获得迁出30天北京的百度迁徙平均值，生成means_beijin列表
plt.plot(np.arange(len(des)),means_beijing)#绘制迁出30天北京的百度迁徙平均值与每一天关系的折线图
print(max(means_beijing),np.argmax(means_beijing))#最大值为0.9014,天数为第3天，即20200103
print(plt.show())

# %%
plt.title('迁入30天北京的百度迁徙平均值')
means_beijing1=[]
for i in range(len(des1)):
    mean=des1[i].loc['mean',:][0]
    means_beijing1.append(mean)#获得迁入30天北京的百度迁徙平均值，生成means_beijin1列表
plt.plot(np.arange(len(des1)),means_beijing1)#绘制迁入30天北京的百度迁徙平均值与每一天关系的折线图
print(max(means_beijing1),np.argmax(means_beijing1))#最大值为0.9214,天数为第13天，即20200113
print(plt.show())
# %%
np.argmax(means_beijing1)

# %%
bar = pyec.Bar()
x=list(des[0].loc['max',:].index)
y=list(des[0].loc['max',:])
bar.add_xaxis(x)
bar.add_yaxis(series_name='2020年1月1号全部城市的百度迁徙迁出最大值',y_axis=y)
bar.set_global_opts(title_opts=opts.TitleOpts(title="标题"))
print(bar.render('1.html'))#绘制2020年1月1号全部城市的百度迁徙迁出最大值与全部城市的关系的饼图，最大值为79.02，城市为克孜勒苏柯尔克孜自治州

# %%
bar = pyec.Bar()
x=list(des1[0].loc['max',:].index)
y=list(des1[0].loc['max',:])
bar.add_xaxis(x)
bar.add_yaxis(series_name='2020年1月1号全部城市的百度迁徙迁入最大值',y_axis=y)
bar.set_global_opts(title_opts=opts.TitleOpts(title="标题"))#绘制2020年1月1号全部城市的百度迁徙迁入最大值与全部城市的关系的饼图，且最大值为76.42，城市为克孜勒苏柯尔克孜自治州
print(bar.render('2.html'))

# %%



