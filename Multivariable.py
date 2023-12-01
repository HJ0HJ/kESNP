#!/usr/bin/env python
# -*- coding:utf-8 -*-
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from sklearn.preprocessing import LabelEncoder

def set_seed(seed=None):
    if seed is None:
        import time
        seed = int((time.time() * 10 ** 6) % 4294967295)
    try:
        np.random.seed(seed)
    except Exception as e:
        print("!!! WARNING !!!: Seed was not set correctly.")
        print("!!! Seed that we tried to use: " + str(seed))
        print("!!! Error message: " + str(e))
        seed = None
    return seed

def normalize(ori_data, flag='01'):
    data = ori_data.copy()
    if len(data.shape) > 1:  # 2-D
        N, K = data.shape
        minV = np.zeros(shape=K)
        maxV = np.zeros(shape=K)
        for i in range(N):
            minV[i] = np.min(data[i, :])
            maxV[i] = np.max(data[i, :])
            if np.abs(maxV[i] - minV[i]) > 0.00001:
                if flag == '01':  # normalize to [0, 1]
                    data[i, :] = (data[i, :] - minV[i]) / (maxV[i] - minV[i])
                else:
                    data[i, :] = 2 * (data[i, :] - minV[i]) / (maxV[i] - minV[i]) - 1
        return data, maxV, minV
    else:  # 1D
        minV = np.min(data)
        maxV = np.max(data)
        if np.abs(maxV - minV) > 0.00001:
            if flag == '01':  # normalize to [0, 1]
                data = (data - minV) / (maxV - minV)
            else:
                data = 2 * (data - minV) / (maxV - minV) - 1
        return data, maxV, minV

def re_normalize(ori_data, maxV, minV, flag='01'):
    data = ori_data.copy()
    if len(data.shape) > 1:  # 2-D
        Nc, K = data.shape
        for i in range(Nc):
            if np.abs(maxV[i] - minV[i]) > 0.00001:
                if flag == '01':  # normalize to [0, 1]
                    data[i, :] = data[i, :] * (maxV[i] - minV[i]) + minV[i]
                else:
                    data[i, :] = (data[i, :] + 1) * (maxV[i] - minV[i]) / 2 + minV[i]
    else:  # 1-D
        if np.abs(maxV - minV) > 0.00001:
            if flag == '01':  # normalize to [0, 1]
                data = data * (maxV - minV) + minV
            else:
                data = (data + 1) * (maxV - minV) / 2 + minV
    return data

import pandas as pd
# dataset 9: PM2.5.csv #todo:  good
pollution = pd.read_csv(r'./dataset/pollution.csv', delimiter=',').values
#pollution = pollution[:, 1:]  # .astype(np.float)
encoder = LabelEncoder()#标签编码器，将离散数据标准化，并在范围内编码
pollution[:, 5] = encoder.fit_transform(pollution[:, 5])#安装标签编码器并返回编码的标签
# float ensure all data is float
pollution = pollution[:, 1:].astype('float32')
dataset0 = np.array(pollution[:,:], dtype=np.float32)
#将原来的dataset0所有值转置
data0 = np.zeros(shape=(dataset0.shape[1], dataset0.shape[0]))
for i in range(8):
    data0[i, :] = dataset0[:, i]
data, maxV, minV = normalize(data0,'-01')
dataset_name = 't9_last'
seed = 42
trainLen = 29345
testLen = 14454
inSize=outSize=8
resSize=62
spectral_radius=0.98
sigma=0.4
input_scaling=0.5
reg=0.0001
initLen=5
_k=np.arange(2,11,1)
α = 0.5
print("α=",α)

# dataset 10: lorenz.csv #todo:  good
#lorenz = pd.read_csv(r'./dataset/lorenz.csv',header=None).values
#dataset0 = lorenz[:, :].astype(np.float)
#data0 = np.zeros(shape=(dataset0.shape[1], dataset0.shape[0]))
#for i in range(3):
#    data0[i, :] = dataset0[:, i]
#data, maxV, minV = normalize(data0,'-01')
#dataset_name = 't10_last'
#seed = 42
#trainLen = 2009
#testLen = 990

# dataset 11: rossler.csv #todo:  good
#rossler = pd.read_csv(r'./dataset/rossler.csv',header=None).values
#dataset0 = rossler[:, :].astype(np.float)
#data0 = np.zeros(shape=(dataset0.shape[1], dataset0.shape[0]))
#for i in range(3):
#    data0[i, :] = dataset0[:, i]
#data, maxV, minV = normalize(data0,'-01')
#dataset_name = 't11_last'
#seed = 42
#trainLen = 3684
#testLen = 1815

def list_of_groups(init_list, children_list_len):
    """ :param init_list: 原列表
        :param children_list_len: 指定切割的子列表的长度（就是你要分成几份）
        :return: 新列表"""
    list_of_groups = zip(*(iter(init_list),) * children_list_len)
    end_list = [list(i) for i in list_of_groups]
    count = len(init_list) % children_list_len
    end_list.append(init_list[-count:]) if count != 0 else end_list
    return end_list

import xlwt
book = xlwt.Workbook()
sheet = book.add_sheet('sheet1',cell_overwrite_ok='True')
data1 = []
import numpy as np
mode = 'prediction'
import math
min_rmse = math.inf
RMSE_V = 0
MAE_V = 0
MSE_V = 0
RMSE_std = []
MSE_std = []
MAE_std = []
set_seed(seed)
Win = (np.random.rand(resSize, 1 + inSize) - 0.5) * input_scaling
W = np.random.rand(resSize, resSize) - 0.5
u = np.zeros((resSize, 1))
U = np.zeros((2, resSize))
Z = np.random.binomial(1, 1 - sigma, (
        resSize, resSize))  # Zero-one matrix with density 1-alpha (sparsity alpha)
W = np.multiply(W, Z)  # Element-wise multiply to get desired sparsity
rhoW = max(abs(linalg.eig(W)[0]))
W *= spectral_radius / rhoW
H = np.zeros((1 + inSize + resSize, trainLen - initLen))
Yt = data[:, initLen + 1:trainLen + 1]  # Multivariable
for k in _k:
    for i in range(30):
        Wstate = np.random.rand(resSize, k) - 0.5
        for t in range(trainLen):
            x = data[:, t]  # Multivariable
            # ESNP update equation
            u = α * u + np.diagonal(
                np.dot(W, np.tanh(np.dot(Win, np.r_[np.ones(1), x]).reshape(resSize, 1) + np.dot(Wstate, U)))).reshape(
                resSize, 1)
            U = np.r_[U, u.T]
            U = U[0:k, :]
            if t >= initLen:
                H[:, t - initLen] = np.r_[
                    (np.ones(1 + inSize - 8), x, u.reshape(resSize, ))]  # Multivariable
        H_T = H.T
        if reg is not None:
            # use ridge regression
            Wout = np.dot(np.dot(Yt, H_T), linalg.inv(np.dot(H, H_T) + \
                                                      reg * np.eye(1 + inSize + resSize)))
        else:
            # use pseudo inverse
            Wout = np.dot(Yt, linalg.pinv(H))
        Y = np.zeros((outSize, testLen))
        x = data[:, trainLen]  # Multivariable
        for t in range(testLen):
            u = α * u + np.diagonal(
                np.dot(W, np.tanh(np.dot(Win, np.r_[np.ones(1), x]).reshape(resSize, 1) + np.dot(Wstate, U)))).reshape(
                resSize, 1)
            U = np.r_[U, u.T]
            U = U[0:k, :]
            y = np.dot(Wout, np.r_[np.ones(1), x, u.reshape(resSize, )])  # Multivariable
            Y[:, t] = y
            if mode == 'generative':
                x = y
            elif mode == 'prediction':
                x = data[:, trainLen + t + 1]  # Multivariable
        # compute MSE,RMSE,MAE for the first errorLen time stepsz
        errorLen = testLen
        np.savetxt('./predict/' + dataset_name + str(k) + '.csv', Y[0, 0:errorLen], delimiter=',')
        #Y = re_normalize(Y, maxV, minV, '-01')  # Multivariable
        mse = sum(np.square(data[0, trainLen + 1:trainLen + errorLen + 1] - Y[0, 0:errorLen])) / errorLen
        #mse = sum(np.square(data0[0,trainLen + 1:trainLen + errorLen + 1] - Y[0,0:errorLen])) / errorLen
        MSE_std.append(mse)
        from sklearn.metrics import mean_squared_error
        from math import sqrt

        rmse = sqrt(mean_squared_error(data[0, trainLen + 1:trainLen + errorLen + 1],Y[0, 0:errorLen]))
        #rmse = sqrt(mean_squared_error(data0[0,trainLen + 1:trainLen + errorLen + 1], Y[0,0:errorLen]))
        RMSE_std.append(rmse)
        from sklearn.metrics import median_absolute_error

        mae = median_absolute_error(data[0, trainLen + 1:trainLen + errorLen + 1],Y[0, 0:errorLen])
        #mae = median_absolute_error(data0[0, trainLen + 1:trainLen + errorLen + 1], Y[0, 0:errorLen])
        MAE_std.append(mae)
        # mae = median_absolute_error(data_org, data_pred)
        MAE_V += mae
        MSE_V += mse
        RMSE_V += rmse
        print('k=', k)
        print('RMSE=' + str(RMSE_V / 30) + '±' + str(np.std(RMSE_std)))
        print('MAE=' + str(MAE_V / 30) + '±' + str(np.std(MAE_std)))
        print('MSE=' + str(MSE_V / 30) + '±' + str(np.std(MSE_std)))
    U = np.r_[U, u.T]
    U = U[0:k + 1, :]
    print('k=', k)
    print('RMSE=' + str(RMSE_V / 30) + '±' + str(np.std(RMSE_std)))
    print('MAE=' + str(MAE_V / 30) + '±' + str(np.std(MAE_std)))
    print('MSE=' + str(MSE_V / 30) + '±' + str(np.std(MSE_std)))
    RMSE = str(RMSE_V / 30)
    MAE = str(MAE_V / 30)
    MSE = str(MSE_V / 30)
    if (RMSE_V / 30) < min_rmse:
        min_rmse = RMSE_V / 30
        resSize_N = resSize
    print('resSize_N=' + str(resSize_N))
    data1.append(k)
    data1.append(RMSE)
    data1.append(MAE)
    data1.append(MSE)
    Y = re_normalize(Y, maxV, minV, '-01')
    fig4 = plt.figure()
    ax41 = fig4.add_subplot(111)
    time = range(testLen)
    ax41.plot(time, data0[0, trainLen + 1:trainLen + testLen + 1], 'r-', label='the original data')
    ax41.plot(time, Y[0, 0:errorLen], 'g--', label='the predicted data')
    ax41.set_ylabel("Magnitude")
    ax41.set_xlabel('Time')
    ax41.set_title('PM2.5')
    plt.savefig('./picture/' + 'PM2.5' + str(k) + '.png')
    fig5 = plt.figure()
    ax5 = fig5.add_subplot(111)
    ax5.plot(time, np.abs(data0[0, trainLen + 1:trainLen + testLen + 1] - Y[0, 0:errorLen]), 'r')
    ax5.set_ylabel("Magnitude")
    ax5.set_xlabel('Time')
    ax5.set_title('PM2.5')
    plt.savefig('./picture/' + '1PM2.5' + str(k) + '.png')
    ax41.legend()
    plt.tight_layout()
print('******************************************')
print('resSize=', resSize)
print('inSize=', inSize)
print('outSize=', outSize)
print('spectral_radius=', spectral_radius)
print('sigma=', sigma)
print('input_scaling=', input_scaling)
print('reg=', reg)
print('initLen=', initLen)
#plt.show()
data1 = list_of_groups(data1, 4)
print("data1=", data1)
import pandas as pd
df=pd.DataFrame(data1,columns=['k','RMSE','MAE','MSE'])
df.to_excel('./result1/PM2.5对应k的结果.xlsx')

