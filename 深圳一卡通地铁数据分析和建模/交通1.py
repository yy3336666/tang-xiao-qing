import matplotlib.pyplot as plt
import pandas as pd
df=pd.read_csv('深圳一卡通刷卡数据\shenzhenkatong1.csv')
df1=pd.read_csv('深圳一卡通刷卡数据\shenzhenkatong2.csv')
plt.rcParams['font.sans-serif'] = ['SimHei']
#导入数据
df3=pd.concat([df,df1])#数据按时间合并
df3.dropna(subset='station',inplace=True)#删除car_no

df3['deal_date']=df3['deal_date'].apply(lambda x:'2018/9/1 '+x[9:]+':00')
df3['deal_date']=pd.to_datetime(df3['deal_date'])#对时间序列进行格式转化。
p=df3.sort_values('deal_date').reset_index(drop=True)

a=p['deal_date'].value_counts().sort_index()
plt.plot(a.index,a)
plt.show()
q=df3.groupby('deal_type')['deal_date'].value_counts()
plt.plot(q['地铁入站'].sort_index().index,q['地铁入站'].sort_index())
plt.plot(q['地铁出站'].sort_index().index,q['地铁出站'].sort_index())#绘制不同线路勾勒出曲线图
r=df3.groupby('station')['deal_date'].value_counts()
for i in df3['station'].unique()[:10]:
    plt.plot(r[i].sort_index().index,r[i].sort_index())
plt.show()
print(df3['station'].unique()[:10])#绘制不同站点勾勒出曲线图