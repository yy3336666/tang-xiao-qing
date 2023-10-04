import pandas as pd
import numpy as np
import jieba 
from collections import Counter
def get_index(lst=None, item=''):
    return [i for i in range(len(lst)) if lst[i] == item] 
with open("stoplist.txt", encoding="utf-8-sig") as f:#打开停用词
    stop_words = f.read().split()
stop_words = set(stop_words)
with open('content_1673171156399.txt',encoding='utf-8') as f:#打开文件
    number={}
    a=[]
    b=[]
    c=[]
    for line in f.readlines():
        print(line)
        if line.startswith('作者'):
            line=line.replace('作者:','')#将作者字符添加到列表
            line=line.strip()
            a.append(line)
        if line.startswith('诗体'):
            line=line.replace('诗体:','')#将诗体字符添加到列表
            line=line.strip()
            b.append(line)
        if line.startswith('诗文'):
            line=line.replace('诗文:','')#将诗文字符添加到列表
            count1=jieba.lcut(line)
            g=Counter(count1)
            del g['，']
            del g['。']
            del g['(']
            c.append(g)
    count=Counter(a)
print
d=[]
for i in set(a):
    e={}
    index=get_index(a,i)
    for j in index:
        e.update(c[j])
    d.append((i,sorted(e.items(),key=lambda x:x[1],reverse=True)[0][0]))#生成每个诗人的最高频率词
print(d)
f=[]
for i in set(b):
    e={}
    index=get_index(b,i)
    for j in index:
        e.update(c[j])
    f.append((i,sorted(e.items(),key=lambda x:x[1],reverse=True)[0][0]))#生成每个诗的类型的最高频率词汇
print(f)