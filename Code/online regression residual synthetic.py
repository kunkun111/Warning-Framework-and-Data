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
# from mpl_tookits.mplot3d import Axes3D




'''
# Generate incremental data

data = np.zeros((200, 6))
for i in range (200):
    
    m = i % 50
    n = i // 50
    
    x1 = np.random.uniform(0,1)
    x2 = np.random.uniform(0,2)
    # x3 = np.random.uniform(0,3)
    if i<=120:
        x4 = 1
    else:
        x4 = 0
    x5 = np.random.uniform(0,1)
    
    
    if n == 0:
        # x1 = i*0.1 + np.random.uniform(0,1)
        # x2 = i*0.1 + np.random.uniform(0,2)
        x3 = i*0.1 + np.random.uniform(0,3)
        data[i,5] = i*0.1 +  x1+x2
    if n == 1:
        # x1 = 50*0.1 * n + np.random.uniform(0,1)
        # x2 = 50*0.1 * n + np.random.uniform(0,2)
        x3 = 50*0.1 * n + np.random.uniform(0,3)
        data[i,5] = 50*0.1 * n + x1+x2
    if n == 2:
        # x1 = 150*0.1 - i*0.1 + np.random.uniform(0,1)
        # x2 = 150*0.1 - i*0.1 + np.random.uniform(0,2)
        x3 = 150*0.1 - i*0.1 + np.random.uniform(0,3)
        data[i,5] = 150*0.1 - i*0.1 + x1+x2
    if n == 3:
        # x1 =  np.random.uniform(0,1)
        # x2 =  np.random.uniform(0,2)
        x3 =  np.random.uniform(0,3)
        data[i,5] = x1+x2
    
    data[i,0] = x1
    data[i,1] = x2
    data[i,2] = x3
    data[i,3] = x4
    data[i,4] = x5
        
plt.figure(figsize=(8,3))    
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.plot(data[:,5], label = '增量型事件状态')
plt.axvline(50, color = 'Orange', linestyle = '--')
plt.axvline(100, color = 'Orange',  linestyle = '--')
plt.axvline(150, color = 'Orange', linestyle = '--')
plt.xlabel('时刻')
plt.ylabel('事件状态')
plt.ylim(0, 20)
plt.legend(loc = 'upper right')
plt.show()

np.savetxt('data_incremental.csv', data, delimiter = ',')




# Generate sudden data
data = np.zeros((200, 6))
for i in range (200):
    
    m = i % 50
    n = i // 50
    
    x1 = np.random.uniform(0,1)
    x2 = np.random.uniform(0,2)
    x3 = np.random.uniform(0,3)
    if i<=120:
        x4 = 1
    # elif i<=600:
    #     x4 = 1
    else:
        x4 = 0
    x5 = np.random.uniform(0,1)

    
    if n <= 2:
        # x1 = 50*0.1 * n + np.random.uniform(0,1)
        # x2 = 50*0.1 * n + np.random.uniform(0,2)
        # x3 = 50*0.1 * n + np.random.uniform(0,3)
        data[i,5] = 50*0.1 * n + x1+x2
    if n == 3:
        # x1 = 50*0.1 * (n-2) + np.random.uniform(0,1)
        # x2 = 50*0.1 * (n-2) + np.random.uniform(0,2)
        # x3 = 50*0.1 * (n-2) + np.random.uniform(0,3)
        data[i,5] = 50*0.1 * (n-2) + x1+x2
        
    
    data[i,0] = x1
    data[i,1] = x2
    data[i,2] = x3
    data[i,3] = x4
    data[i,4] = x5
        
plt.figure(figsize=(8,3))    
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.plot(data[:,5], label = '突发型事件状态')
plt.axvline(50, color = 'Orange', linestyle = '--')
plt.axvline(100, color = 'Orange', linestyle = '--')
plt.axvline(150, color = 'Orange', linestyle = '--')
plt.xlabel('时刻')
plt.ylabel('事件状态')
plt.ylim(0, 20)
plt.legend(loc = 'upper right')
plt.show()

np.savetxt('data_sudden.csv', data, delimiter = ',')



# correlation
data = pd.read_csv('.../data_incremental.csv',  names=['x1', 'x2', 'x3', 'x4', 'x5', 'y'])


a = data.iloc[:,:].corr()
plt.subplots(figsize = (9,9))
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
ax = sns.heatmap(a, annot = True, vmax = 1, square = True, cmap = 'Blues', annot_kws = {'fontsize': 15})
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize = 12)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.show()


data = pd.read_csv('.../data_sudden.csv',  names=['x1', 'x2', 'x3', 'x4', 'x5', 'y'])


a = data.iloc[:,:].corr()
plt.subplots(figsize = (9,9))
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
ax = sns.heatmap(a, annot = True, vmax = 1, square = True, cmap = 'Blues', annot_kws = {'fontsize': 15})
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize = 12)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.show()
'''


# (1) online learning

data = pd.read_csv('.../data_incremental.csv',  names=['x1', 'x2', 'x3', 'x4', 'x5', 'y'])
# data = pd.read_csv('.../data_sudden.csv',  names=['x1', 'x2', 'x3', 'x4', 'x5', 'y'])
data = data.values

x_train = data[0:1, :-1]
y_train = data[0:1, -1]

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


stream = data[1:, :]
results = []
final_results = []
loss = []
loss_ini = 0

for i in range (stream.shape[0]):
    
    x_t = stream[i, :-1]
    x_test = x_t.reshape(1, 5)
    y_test = stream[i, -1]
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
        sq_loss = np.square(final_pred - y_test[0])
        # sq_loss = np.absolute(final_pred - y_test[0])
    
    else:
        
        final_results.append(final_pred_ini)
        sq_loss = np.square(final_pred_ini - y_test[0])
        # sq_loss = np.absolute(final_pred_ini - y_test[0])
    
    loss.append(sq_loss)
    loss_final = np.mean(loss)
    loss_final = np.absolute(loss_final)
    
    
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
loss_OR = np.array(loss)  
a = data[:, -1]
b = np.array(final_results)


plt.figure(figsize=(8,3))    
plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.plot(data[:,5], label = '风险信息状态突然变化')
plt.plot(data[:,5], label = '风险信息状态逐渐变化')
plt.axvline(50, color = 'Orange', linestyle = '--')
plt.axvline(100, color = 'Orange', linestyle = '--')
plt.axvline(150, color = 'Orange', linestyle = '--')
plt.xlabel('时刻')
plt.ylabel('状态')
plt.ylim(0, 20)
plt.legend(loc = 'upper right')
plt.show()


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.figure(figsize=(8,3))   
plt.plot(a, label = '目标值', c = 'b')
plt.plot(b, label = '预测值', c = 'r', linestyle = '--')
plt.xlabel('时刻')
plt.ylabel('状态')
plt.ylim(0, 20)
plt.legend(loc = 'upper left')
plt.show()

# original data
plt.rcParams['figure.figsize']=(9, 4)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False 
ax1 = plt.subplot(111, projection = '3d')
xx = np.arange(0,199,1)
yy = data[1:,4] 
zz = data[1:,-1]

ax1.scatter(xx, yy, zz, c = 'b', s = 2, marker = '*', label = '状态')

ax1.set_xlabel('时刻')
ax1.set_xlim(1,200)
ax1.set_ylabel('噪声')
ax1.set_ylim(-3,3)
ax1.set_zlabel('状态')
ax1.legend()
plt.show()

# experiment results
plt.rcParams['figure.figsize']=(9, 4)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False 
ax1 = plt.subplot(111, projection = '3d')
xx = np.arange(0,199,1)
yy = data[1:,4] 
zz = data[1:,-1]
pp = np.array(final_results)

ax1.scatter(xx, yy, zz, c = 'b', s = 2, marker = '*', label = '目标值')
ax1.scatter(xx, yy, pp, c = 'r', s = 2, marker = 'v', label = '预测值')

ax1.set_xlabel('时刻')
ax1.set_xlim(1,200)
ax1.set_ylabel('噪声')
ax1.set_ylim(-3,3)
ax1.set_zlabel('状态')
ax1.legend()
plt.show()


# (2) feature selection

data = pd.read_csv('.../data_incremental.csv',  names=['x1', 'x2', 'x3', 'x4', 'x5', 'y'])
# data = pd.read_csv('.../data_sudden.csv',  names=['x1', 'x2', 'x3', 'x4', 'x5', 'y'])
data = data.values

data = np.delete(data, 3, axis = 1)

x_train = data[0:1, :-1]
y_train = data[0:1, -1]

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


stream = data[1:, :]
results = []
final_results = []
loss_f = []
loss_ini = 0
f_iter1 = []
f_iter2 = []
idex = 0

for i in range (stream.shape[0]):
    
    x_t = stream[i, :-1]
    x_test = x_t.reshape(1, 4)
    y_test = stream[i, -1]
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
    
    else:
        
        final_results.append(final_pred_ini)
        sq_loss_f = np.square(final_pred_ini - y_test[0])
        # sq_loss_f = np.absolute(final_pred_ini - y_test[0])

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
    
a = data[:, -1]
b = np.array(final_results)
plt.rcParams['font.sans-serif'] = ['SimHei']

plt.figure(figsize=(8,3))   
plt.plot(a, label = '目标值', c = 'b')
plt.plot(b, label = '预测值', c = 'r', linestyle = '--')


plt.xlabel('时刻')
plt.ylabel('状态')
plt.ylim(0, 20)
plt.legend(loc = 'upper left')
plt.show()



# feature iter
it1 = np.array(f_iter1)
it2 = np.array(f_iter2)
plt.rcParams['font.sans-serif'] = ['SimHei']

plt.figure(figsize=(8,3))   
plt.scatter(it1[:,1], it1[:,0], s = 10, marker = 'o', label = '选择特征x4')
plt.scatter(it2[:,1], it2[:,0], s = 10, marker = 'v', label = '不选择特征x4')
plt.xlabel('时刻')
plt.ylabel('特征选择')
plt.ylim(-1, 2)
plt.legend(loc = 'upper left')
plt.show()



# (3) potential prediction
'''
# data generation
data = np.zeros((230, 6))
for i in range (230):
    
    m = i % 50
    n = i // 50
    
    x1 = np.random.uniform(0,1)
    x2 = np.random.uniform(0,2)
    # x3 = np.random.uniform(0,3)
    if i<=120:
        x4 = 1
    # elif i<=600:
    #     x4 = 1
    else:
        x4 = 0
    x5 = np.random.uniform(0,1)
    
    
    if n == 0:
        # x1 = i*0.1 + np.random.uniform(0,1)
        # x2 = i*0.1 + np.random.uniform(0,2)
        x3 = i*0.1 + np.random.uniform(0,3)
        data[i,5] = i*0.1 +  x1+x2
    if n == 1:
        # x1 = 50*0.1 * n + np.random.uniform(0,1)
        # x2 = 50*0.1 * n + np.random.uniform(0,2)
        x3 = 50*0.1 * n + np.random.uniform(0,3)
        data[i,5] = 50*0.1 * n + x1+x2
    if n == 2:
        # x1 = 150*0.1 - i*0.1 + np.random.uniform(0,1)
        # x2 = 150*0.1 - i*0.1 + np.random.uniform(0,2)
        x3 = 150*0.1 - i*0.1 + np.random.uniform(0,3)
        data[i,5] = 150*0.1 - i*0.1 + x1+x2
    if n == 3:
        # x1 =  np.random.uniform(0,1)
        # x2 =  np.random.uniform(0,2)
        x3 =  np.random.uniform(0,3)
        data[i,5] = x1+x2
    if n == 4:
        # x1 =  np.random.uniform(0,1)
        # x2 =  np.random.uniform(0,2)
        x3 =  np.random.uniform(0,3)
        data[i,5] = i*0.1 - 200*0.1 +x1+x2
    
    data[i,0] = x1
    data[i,1] = x2
    data[i,2] = x3
    data[i,3] = x4
    data[i,4] = x5
        
plt.figure(figsize=(8,3)) 
plt.rcParams['font.sans-serif'] = ['SimHei']   
plt.plot(data[:,5], label = '（原始）增量型事件状态')
plt.plot(data[10:,5], label = '（调整后）增量型事件状态')
plt.xlabel('时刻')
plt.ylabel('事件状态')
plt.ylim(0, 20)

plt.axvline(50, color = 'Orange', linestyle = '--')
plt.axvline(100, color = 'Orange', linestyle = '--')
plt.axvline(150, color = 'Orange', linestyle = '--')
plt.axvline(200, color = 'Orange', linestyle = '--')

plt.legend(loc = 'upper right')
plt.show()

np.savetxt('data_incremental_potential.csv', data, delimiter = ',')




# Generate sudden data
data = np.zeros((230, 6))
for i in range (230):
    
    m = i % 50
    n = i // 50
    
    x1 = np.random.uniform(0,1)
    x2 = np.random.uniform(0,2)
    x3 = np.random.uniform(0,3)
    if i<=120:
        x4 = 1
    # elif i<=600:
    #     x4 = 1
    else:
        x4 = 0
    x5 = np.random.uniform(0,1)

    
    if n <= 2:
        # x1 = 50*0.1 * n + np.random.uniform(0,1)
        # x2 = 50*0.1 * n + np.random.uniform(0,2)
        # x3 = 50*0.1 * n + np.random.uniform(0,3)
        data[i,5] = 50*0.1 * n + x1+x2
    if n == 3:
        # x1 = 50*0.1 * (n-2) + np.random.uniform(0,1)
        # x2 = 50*0.1 * (n-2) + np.random.uniform(0,2)
        # x3 = 50*0.1 * (n-2) + np.random.uniform(0,3)
        data[i,5] = 50*0.1 * (n-2) + x1+x2
    if n == 4:
        # x1 = 50*0.1 * (n-2) + np.random.uniform(0,1)
        # x2 = 50*0.1 * (n-2) + np.random.uniform(0,2)
        # x3 = 50*0.1 * (n-2) + np.random.uniform(0,3)
        data[i,5] = 50*0.1 * (n-2) + x1+x2
        
    
    data[i,0] = x1
    data[i,1] = x2
    data[i,2] = x3
    data[i,3] = x4
    data[i,4] = x5
        
plt.figure(figsize=(8,3))    
plt.plot(data[:,5], label = '（原始）突发型事件状态')
plt.plot(data[10:,5], label = '（调整后）突发事件状态')
plt.xlabel('时刻')
plt.ylabel('事件状态')
plt.ylim(0, 20)

plt.axvline(50, color = 'Orange', linestyle = '--')
plt.axvline(100, color = 'Orange', linestyle = '--')
plt.axvline(150, color = 'Orange', linestyle = '--')
plt.axvline(200, color = 'Orange', linestyle = '--')

plt.legend(loc = 'upper right')
plt.show()

np.savetxt('data_sudden_potential.csv', data, delimiter = ',')
'''


# online learning
data = pd.read_csv('.../data_incremental_potential.csv',  names=['x1', 'x2', 'x3', 'x4', 'x5', 'y'])
# data = pd.read_csv('.../data_sudden_potential.csv',  names=['x1', 'x2', 'x3', 'x4', 'x5', 'y'])
data = data.values

x = data[0:220, :-1]
y = data[10:230, -1]

x_train = x[0:1, :]
y_train = y[0:1]


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
    x_test = x_t.reshape(1, 5)
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
        sq_loss = np.square(final_pred - y_test[0])
        # sq_loss = np.absolute(final_pred - y_test[0])
        loss.append(sq_loss)
    
    else:
        
        final_results.append(final_pred_ini)
        sq_loss = np.square(final_pred_ini - y_test[0])
        # sq_loss = np.absolute(final_pred_ini - y_test[0])
        loss.append(sq_loss)
    
    
    loss_final = np.mean(loss)
    loss_final = np.absolute(loss_final)
    
    
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
    
  
# print(results)
print('loss:', loss_final)
    
a = data[:, -1]
b = data[10:, -1]
c = np.array(final_results)

plt.rcParams['font.sans-serif'] = ['SimHei']

plt.figure(figsize=(8,3))   
plt.plot(a, label = '目标值（原始）', linestyle = '--')
plt.plot(b, label = '目标值（调整后）')
plt.plot(c, label = '预测值', linestyle = '-.')


plt.xlabel('时刻')
plt.ylabel('状态')
plt.ylim(0, 20)
plt.legend(loc = 'upper left')
plt.show()


# experiment results
plt.rcParams['figure.figsize']=(9, 4)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False 
ax1 = plt.subplot(111, projection = '3d')
xx1 = np.arange(0,229,1)
yy1 = data[1:,4] 
zz1 = data[1:,-1]
ax1.scatter(xx1, yy1, zz1, c = 'b', s = 2, marker = '*', label = '目标值(原始)')

xx2 = np.arange(0,219,1)
yy2 = data[11:,4] 
zz2 = data[11:,-1]
ax1.scatter(xx2, yy2, zz2, c = 'orange', s = 2, marker = 'o', label = '目标值(调整后)')

xx3 = np.arange(0,219,1)
yy3 = data[11:,4] 
zz3 = np.array(final_results)
ax1.scatter(xx3, yy3, zz3, c = 'green', s = 2, marker = 'v', label = '预测值')

ax1.set_xlabel('时刻')
# ax1.set_xlim(1,250)
ax1.set_ylabel('噪声')
ax1.set_ylim(-3,3)
ax1.set_zlabel('状态')
ax1.legend()
plt.show()



# feature selection

data = pd.read_csv('.../data_incremental_potential.csv',  names=['x1', 'x2', 'x3', 'x4', 'x5', 'y'])
# data = pd.read_csv('.../data_sudden_potential.csv',  names=['x1', 'x2', 'x3', 'x4', 'x5', 'y'])
data = data.values

data = np.delete(data, 3, axis = 1)

x = data[0:220, :-1]
y = data[10:230, -1]

x_train = x[0:1, :]
y_train = y[0:1]

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

for i in range (stream_x.shape[0]):
    
    x_t = stream_x[i, :]
    x_test = x_t.reshape(1, 4)
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
    
    else:
        
        final_results.append(final_pred_ini)
        sq_loss_f = np.square(final_pred_ini - y_test[0])
        sq_loss_f = np.absolute(final_pred_ini - y_test[0])

    # feature selection
    if sq_loss_f < loss[i]:
        loss_f.append(sq_loss_f)
        # print(1)
    else:
        loss_f.append(loss[i])
        # print(0)
    
    loss_final = np.mean(loss_f)
    
    
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
    

    
# print(results)
print('loss_f:', loss_final)
    
a = data[:, -1]
b = data[10:, -1]
c = np.array(final_results)

plt.rcParams['font.sans-serif'] = ['SimHei']

plt.figure(figsize=(8,3)) 
plt.plot(a, label = '目标值（原始）', linestyle = '--')
plt.plot(b, label = '目标值（调整后）')
plt.plot(c, label = '预测值', linestyle = '-.')


plt.xlabel('时刻')
plt.ylabel('状态')
plt.ylim(0, 20)
plt.legend(loc = 'upper left')
plt.show()


