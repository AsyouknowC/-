# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 15:40:39 2019

@author: AsyouknowC

"""

import math
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split 
from sklearn.cluster import KMeans

cancer = datasets.load_breast_cancer()

class cancerdata:
    
    data_tmp = cancer.data
    data = pd.DataFrame(data_tmp)
    target = cancer.target
    feature = cancer.feature_names
    tragetname = cancer.target_names
    lendata = len(data)
    lenonedata = len(data[0])
    kind = max(target)+1

x_train,x_test,y_train,y_test = train_test_split(cancerdata.data,cancerdata.target,test_size=0.3,random_state = 0)

def istring(x,string):
    for dot in string:
        if (dot == x):
            return 1
    return 0

#计算熵h(d)
def gethd(lista):
    num = []
    sums = len(lista)
    string = []
    for i in range(sums):
        if i == 0:
            string.append(lista[i])
            num.append(1)
        else:
            if istring(lista[i],string) == 0:
                string.append(lista[i])
                num.append(1)
            else:
                num[string.index(lista[i])] += 1
    result = 0
    for i in range(len(num)):
        result -= (num[i]/sums)*math.log((num[i]/sums),2)
    
    return result

#计算增益熵g(d,a)
def getgd(lista,target,hd):
    bc=0 #差值
    num=[]#按属性
    sums = len(lista)
    string = []
    for i in range(sums):
        if i == 0:
            string.append(lista[i])
            num.append(1)
        else:
            if istring(lista[i],string) == 0:
                string.append(lista[i])
                num.append(1)
            else:
                num[string.index(lista[i])] += 1
    
    
    for i in range(len(num)):
        listb=[]#标签
        for j in range(sums):
            if(lista[j] == i):
                listb.append(target[j])
        
        smallresult = gethd(listb)
        bc += (num[i]/sums)*smallresult
    
    result = hd - bc
    return result

def isame(lists):
    for i in range(len(lists)-1):
        if lists[i] != lists[i+1]:
            return 0
    return 1

def listproduce(data,index):
    lista=[]
    for i in range(len(data)):
        lista.append(data[i][index])
    return lista

#本例中feature默认为5，因为离散化处理中所有属性均被分为5类
def getoptresult(lista,target,feature_kind,target_kind):
    print('getoptresult')
    #第i行第j列代表 在属性为i，标签为j 出现的次数
    #矩阵每行取最大，求其所占该行比例，所有比例相比，中最大数的所在行列即为所求
    '''
    如
    属性 是：0，否：1；标签 是：0，否：1
    是 否
    否 否
    是 是
    是 否
    否 否
    是 是
    
    2,2
    0,2
    
    '''
    feature_mat = np.zeros([feature_kind,target_kind],int)
    
    for i in range(len(lista)):
        feature_mat[lista[i]][target[i]] += 1
    print(feature_mat)
    max_list = []#每行取最大，求其所占该行比例
    max_locat =[]#该最大数值所在位置
    
    for i in range(feature_kind):
        sums= sum(feature_mat[i])
        tmp_list=[]#这一行中每个数所占比例
        for j in range(target_kind):
            tmp_list.append(feature_mat[i][j]/sums)
        
        max_h = max(tmp_list)#此行最大值
        max_list.append(max_h)
        locatcol = tmp_list.index(max_h)
        
        max_locat.append([i,locatcol])
    
    
    max_m = max(max_list)
    max_located = max_locat[max_list.index(max_m)]
    feature = max_located[0]#最大值所在行
    result = max_located[1]#最大值所在列
    return feature,result#feature为可划分的最优属性值，result为最优属性值对应的标签值

def train(x_train,y_train):
    print('trainning')
    list_optimal=[]#最优属性序列
    list_optfeature =[]#最优序列对应的属性值
    list_optresult = [] #最优序列对应属性值的标签
    traincounter = 0#寻找次数 即最优序列个数
    
    newdata = x_train
    newtarget = y_train
    
    while True:
        #计算增益熵，选择划分最优属性
        hd = gethd(newtarget)
        maxculomn =[]# 暂时存储所有的增益熵值
        for i in range(len(newdata[0])):
            listdata = listproduce(newdata,i)#第i列属性值，即所有i属性的值，与newtarget同样一一对应
            tmp = (getgd(listdata,newtarget,hd))
            
            maxculomn.append(tmp)
        
        maxc = maxculomn.index(max(maxculomn))
        list_optimal.append(maxc)
        
        listmax = listproduce(newdata,maxc)#增益熵值最大的属性列
        
        optfeature,optresult = getoptresult(listmax,newtarget,3,cancerdata.kind)
        list_optfeature.append(optfeature)#可划分属性对应的值
        list_optresult.append(optresult)#可划分属性值对应的标签值
        
        #更新数据，去除已经划分的数据
        listdelete = []#需要删除的行
        #需要删除的列为 maxc 最优属性列
        
        for i in range(len(newdata)):
            if newtarget[i] == optresult and newdata[i][maxc] == optfeature:
                listdelete.append(i)
                
        newdata = np.delete(newdata,listdelete,axis = 0)#删除数据行        
        newtarget = np.delete(newtarget,listdelete,axis = 0)#删除标签行
        newdata = np.delete(newdata,maxc,axis = 1)#删除数据列
    
        print('第',traincounter,'次训练,最大增益熵属性列（去除前一次）',maxc)
        print('剩余target长度',len(newtarget))
        print('剩余data长度',len(newdata))
        print(list_optimal)
        print(list_optfeature)
        print(list_optresult)
                
        traincounter += 1
        #当剩余数据所有标签值不可再被划分时或者属性数目小于2时 或决策树深度超过16层时，跳出循环
        if isame(newtarget) == 1 or len(newdata[0])<2 or traincounter > 15:
            break
    
    #构建决策树：
    tree={0:list_optimal,1:list_optfeature,2:list_optresult}
    print('train finished')
    return tree


'''
{'有工作':{是:0,否:{'有自己的房子':{是:1,否:0}}}}

def creatree(list_optimal,list_optfeature,list_optresult):
    
    others = 1 - list_optresult
    deep = len(list_optimal)
    if deep == 1:
        tree[list_optimal[0]]={list_optfeature:list_optresult,'other':others}
    else:
        
    for i in range(len(listoptimal)):
        
        
    
    
    return 0
'''

#数据离散化函数
def discrete(lista):
    #等频
    lista = lista.values.reshape((lista.index.size,1))
    k = 3 #划分的属性种类数每一类相同，均为3
    k_model = KMeans(n_clusters = k,n_jobs = 3)
    result = k_model.fit_predict(lista)
    '''
    k = 5
    result = pd.cut(lista,k,labels=range(5))
    '''
    return result


#对每一列属性进行离散化，预处理数据
def ini_data(data):
    n = len(data)#样本数
    
    n_feature = 30#属性数
    #data n569行n_feature30列
    #注意表格是按列数 先列后行！！！列数*行数
    table = pd.DataFrame(data) #30列*569行
    
    copetable=[] #离散化处理后的表格
    for i in range(n_feature):
        tmp = discrete(table[i])
        copetable.append(tmp)
    
    newtablepose = np.array(copetable) #30行*569列
    
    resultdata = np.transpose(newtablepose)
    return resultdata#569行*30列

#判断当前样本类别
def judge(x_testi,list_optimal,list_optfeature,list_optresult):
    result = 1 - list_optresult[-1]#认为与最后一项不同的结果
    tmp_result = list_optresult
    for i in range(len(list_optimal)):
        if x_testi[i] == list_optfeature[i]:
            result = tmp_result[i]
            break
        else:
            np.delete(x_testi,i,axis = 0)#删除第i行
    return result

def prediction(x_test,list_optimal,list_optfeature,list_optresult):
    print('prediction')
    predictlist=[]
    for i in range(len(x_test)):
        predict_i = judge(x_test[i],list_optimal,list_optfeature,list_optresult)
        predictlist.append(predict_i)
       
    return predictlist

def main():
    train_x = ini_data(x_train)
    train_y = y_train
    #决策树
    tree = train(train_x,train_y)
    
    
    test_x = ini_data(x_test)
    test_y = y_test
    print(len(test_y))
    predict_y = prediction(test_x,tree[0],tree[1],tree[2])
    
    count_same = 0
    for i in range(len(test_y)):
        if predict_y[i] == test_y[i]:
            count_same += 1
    correction = count_same/len(test_y)  
    print('测试集准确率 ',correction)
    


main()