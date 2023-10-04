import matplotlib.pyplot as plt
import pandas as pd
df=pd.read_csv('深圳一卡通刷卡数据\shenzhenkatong1.csv')
df1=pd.read_csv('深圳一卡通刷卡数据\shenzhenkatong2.csv')
plt.rcParams['font.sans-serif'] = ['SimHei']
df_all=pd.concat([df,df1])#数据按时间合并
df_all.isna().value_counts()#显示各列缺失值
df_all.dropna(subset='car_no',inplace=True)#删除car_no
df_all['deal_date']=df_all['deal_date'].apply(lambda x:'2018/9/1 '+x[9:]+':00')

df_all['deal_date']=pd.to_datetime(df_all['deal_date'])#对时间序列进行格式转化。
df_all=df_all.sort_values('deal_date').reset_index(drop=True)
df_all=df_all.set_index('deal_date',drop=False).sort_index()
a=df_all['deal_date'].value_counts()
a=a.sort_index()#对数据按时间进行从小到大排序

plt.plot(a.index,a)#绘制整体人流量曲线图
plt.show()
q=df_all.groupby('deal_type')['deal_date'].value_counts()
plt.plot(q['地铁入站'].sort_index().index,q['地铁入站'].sort_index())
plt.plot(q['地铁出站'].sort_index().index,q['地铁出站'].sort_index())#绘制不同线路勾勒出曲线图
plt.show()
r=df_all.groupby('station')['deal_date'].value_counts()
for i in df_all['station'].unique()[:10]:
    plt.plot(r[i].sort_index().index,r[i].sort_index())
plt.show()
print(df_all['station'].unique()[:10])#绘制不同站点勾勒出曲线图