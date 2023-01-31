# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 18:05:30 2022

@author: Administrator
"""


import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn import metrics
from tqdm import tqdm
import matplotlib.pyplot as plt
import arff
from skmultiflow.drift_detection import DDM
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
import time
from matplotlib.font_manager import FontProperties




# correlation
data = pd.read_csv('.../Shanghai2022.csv', names=['前日平均气温', '每日风速情况', '每日降雨情况', '每日最高温度'])

plt.rcParams['font.sans-serif'] = ['SimHei']
a = data.iloc[:,:].corr()
plt.subplots(figsize = (9,9))
plt.rcParams['axes.unicode_minus'] = False
ax = sns.heatmap(a, annot = True, vmax = 1, square = True, cmap = 'Blues', annot_kws = {'fontsize': 15})
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize = 12)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.show()



# data process
data = pd.read_csv('.../Shanghai2022.csv', header = None)
data = data.values
n = data.shape[0]


# online learning
z = 5

x = data[:, :-1]
y = data[:, -1]

x_train = x[0:1, :]
y_train = y[0:1]

# print(x_train, y_train)

T1 = time.time()


# fit initial model
np.random.seed(0)
model = GradientBoostingRegressor()
model.fit(x_train, y_train)


# fit residual
res_ini = np.zeros((1))
res_ini = res_ini.reshape(1,1)

np.random.seed(0)
model_res = GradientBoostingRegressor()
model_res.fit(res_ini, y_train)


stream_x = x[1:, :]
stream_y = y[1:]
results = []
final_results = []
loss = []
loss_ini = 0

for i in range (stream_x.shape[0]):
    
    x_t = stream_x[i, :]
    x_test = x_t.reshape(1, 3)
    y_test = stream_y[i]
    y_test = np.array([y_test])
    
    # print(x_test, y_test)
    
    
    # test the model
    y_pred = model.predict(x_test)
    pred = y_pred[0]
    # print(pred)
    
    
    # calculate the residual and test the model_res
    res = pred - y_test[0]
    res_pred = model_res.predict(res.reshape(1,1))
    r_pred = res_pred[0]
    
    
    # get the final results
    final_pred = pred + r_pred * 0.1
    final_pred_ini = pred
    
    results.append(pred)  
    
    loss_new = final_pred - y_test[0]
    loss_old = final_pred_ini - y_test[0]
    
    if abs(loss_new) <= abs(loss_old):
        
        final_results.append(final_pred)
        sq_loss = np.square(final_pred - y_test[0])
        # sq_loss = np.absolute(final_pred - y_test[0])
        loss.append(sq_loss)
        
        # print(final_pred, y_test[0],'*')
        # print(sq_loss,'**')
    
    else:
        
        final_results.append(final_pred_ini)
        sq_loss = np.square(final_pred_ini - y_test[0])
        # sq_loss = np.absolute(final_pred_ini - y_test[0])
        loss.append(sq_loss)
        
        # print(final_pred_ini, y_test[0],'%')
        # print(sq_loss,'%%')
    
    
    loss_final = np.mean(loss)
    # loss_final = np.absolute(loss_final)
    
    
    # ADWIN-based self-adaptation process
    if loss_final > loss_ini:
        
        # print('0')
        x_train = x_test
        y_train = y_test
        
        res_ini = np.zeros((1))
        res_ini[0] = res
        res_ini = res_ini.reshape(1,1)
        
    else:
        
        # print('1')
        x_train = np.vstack((x_train, x_test))
        y_train = np.hstack((y_train, y_test))
        res_ini = np.vstack((res_ini, res))
    
    np.random.seed(0)
    model = GradientBoostingRegressor()
    model.fit(x_train, y_train)
    
    np.random.seed(0)
    model_res = GradientBoostingRegressor()
    model_res.fit(res_ini, y_train)
    
    loss_ini = loss_final
    
T2 = time.time()
print('程序运行时间:%s秒' % (T2 - T1))  
  
# print(results)
print('loss:', loss_final)
    
a = data[1:, -1]
c = np.array(final_results)

plt.rcParams['font.sans-serif'] = ['SimHei']

plt.figure(figsize=(8,3))   
plt.plot(a, label = '目标值（原始）', linestyle = '--')
# plt.plot(b, label = '目标值（调整后）')
plt.plot(c, label = '预测值')


plt.xlabel('时刻')
plt.ylabel('空气质量指数')
# plt.ylim(0, 20)
plt.legend(loc = 'upper left')
plt.show()




# feature selection


data = np.delete(data, 2, axis = 1)

x = data[:, :-1]
y = data[:, -1]

x_train = x[0:1, :]
y_train = y[0:1]

y_ini = y_train[0]

# fit initial model
np.random.seed(0)
model = GradientBoostingRegressor()
model.fit(x_train, y_train)


# fit residual
res_ini = np.zeros((1))
res_ini = res_ini.reshape(1,1)

np.random.seed(0)
model_res = GradientBoostingRegressor()
model_res.fit(res_ini, y_train)


stream_x = x[1:, :]
stream_y = y[1:]
results = []
final_results = []
loss_f = []
loss_ini = 0
loss_warning = []
f_iter1 = []
f_iter2 = []
idex = 0

for i in range (stream_x.shape[0]):
    
    x_t = stream_x[i, :]
    x_test = x_t.reshape(1, 2)
    y_test = stream_y[i]
    y_test = np.array([y_test])
    
    
    # test the model
    y_pred = model.predict(x_test)
    pred = y_pred[0]
    
    
    # calculate the residual and test the model_res
    res = pred - y_test[0]
    res_pred = model_res.predict(res.reshape(1,1))
    r_pred = res_pred[0]
    
    
    # get the final results
    final_pred = pred + r_pred * 0.1
    final_pred_ini = pred
    
    results.append(pred)
    
    loss_new = final_pred - y_test[0]
    loss_old = final_pred_ini - y_test[0]
    
    if abs(loss_new) <= abs(loss_old):
        
        final_results.append(final_pred)
        sq_loss_f = np.square(final_pred - y_test[0])
        # sq_loss_f = np.absolute(final_pred - y_test[0])
        # loss.append(sq_loss_f)
        k = final_pred
        # print(sq_loss_f)
    
    else:
        
        final_results.append(final_pred_ini)
        sq_loss_f = np.square(final_pred_ini - y_test[0])
        # sq_loss_f = np.absolute(final_pred_ini - y_test[0])
        # loss.append(sq_loss_f)
        k = final_pred_ini
        # print(sq_loss_f)

    # feature selection
    if sq_loss_f < loss[i]:
        loss_f.append(sq_loss_f)
        f_iter1.append((1, idex))
        # print(1)
    else:
        loss_f.append(loss[i])
        f_iter2.append((0, idex))
        # print(0)
    idex += 1
    
    loss_final = np.mean(loss_f)
    
    
    # ADWIN-based self-adaptation process
    if loss_final > loss_ini:
        
        # print('0')
        x_train = x_test
        y_train = y_test
        
        res_ini = np.zeros((1))
        res_ini[0] = res
        res_ini = res_ini.reshape(1,1)
        
        loss_warning.append((i, k))
    else:
        
        # print('1')
        x_train = np.vstack((x_train, x_test))
        y_train = np.hstack((y_train, y_test))
        res_ini = np.vstack((res_ini, res))
    
    np.random.seed(0)
    model = GradientBoostingRegressor()
    model.fit(x_train, y_train)
    
    np.random.seed(0)
    model_res = GradientBoostingRegressor()
    model_res.fit(res_ini, y_train)
    
    
    loss_ini = loss_final
    

    
# print(results)
print('loss_f:', loss_final)
    
a = data[1:, -1]
c = np.array(final_results)

plt.rcParams['font.sans-serif'] = ['SimHei']

plt.figure(figsize=(9,3)) 
plt.plot(a, label = '目标值')
# plt.plot(b, label = '目标值（调整后）')
plt.plot(c, label = '预测值', linestyle = '--', color = 'r')

c = c = np.array(loss_warning)

plt.scatter(c[:,0], c[:,1], marker = 'v', color = 'lime',label = '预警信号')


plt.xlabel('时刻')
# plt.ylabel('单日空气质量指数')
plt.ylabel('单日最高气温(摄氏度)')
# plt.ylim(0, 20)
plt.legend(loc = 'upper left')
plt.show()


# feature iter
it1 = np.array(f_iter1)
it2 = np.array(f_iter2)
plt.rcParams['font.sans-serif'] = ['SimHei']

plt.figure(figsize=(8,3))   
plt.scatter(it1[:,1], it1[:,0], s = 8, marker = 'o', label = '选择每日降雨情况特征')
plt.scatter(it2[:,1], it2[:,0], s = 8, marker = 'v', label = '不选择每日降雨情况特征')
plt.xlabel('时刻')
plt.ylabel('特征选择')
plt.ylim(-1, 2)
plt.legend(loc = 'upper left')
plt.show()























