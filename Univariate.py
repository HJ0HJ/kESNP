import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import math
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import median_absolute_error
import datetime
# normalize the data
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
                if flag == '01':
                    data[i, :] = (data[i, :] - minV[i]) / (maxV[i] - minV[i])
                else:
                    data[i, :] = 2 * (data[i, :] - minV[i]) / (maxV[i] - minV[i]) - 1
        return data, maxV, minV
    else:  # 1D
        minV = np.min(data)
        maxV = np.max(data)
        if np.abs(maxV - minV) > 0.00001:
            if flag == '01':
                data = (data - minV) / (maxV - minV)
            else:
                data = 2 * (data - minV) / (maxV - minV) - 1
        return data, maxV, minV
# renormalize the data
def re_normalize(ori_data, maxV, minV, flag='01'):
    data = ori_data.copy()
    if len(data.shape) > 1:  # 2-D
        Nc, K = data.shape
        for i in range(Nc):
            if np.abs(maxV[i] - minV[i]) > 0.00001:
                if flag == '01':
                    data[i, :] = data[i, :] * (maxV[i] - minV[i]) + minV[i]
                else:
                    data[i, :] = (data[i, :] + 1) * (maxV[i] - minV[i]) / 2 + minV[i]
    else:  # 1-D
        if np.abs(maxV - minV) > 0.00001:
            if flag == '01':
                data = data * (maxV - minV) + minV
            else:
                data = (data + 1) * (maxV - minV) / 2 + minV
    return data

# dataset 1:MG
#import scipy.io as sio
#data = sio.loadmat('./dataset/MG_chaos.mat')['dataset'].flatten()
 # only use data from t=124 : t=1123  (all data previous are not in the same pattern!)
#data0 = data[123:1123]
#data, maxV, minV = normalize(data0,'-01')
#划分数据集并读取数据
#seed = 42  # None
#trainLen = 662
#testLen =329
#dataset_name = 't1_last'

# dataset 2 :sp500
#sp500_src = "./dataset/sp500.csv"
#dateparse = lambda x: datetime.datetime.strptime(x, '%Y-%m-%d')
#sp500 = pd.read_csv(sp500_src, delimiter=',', parse_dates=[0], date_parser=dateparse).values
#data0 = np.array(sp500[:, 1], dtype=np.float32)
#data, maxV, minV = normalize(data0,'-01')
#dataset_name = 't2_last'
# 划分数据集并读取数据
#seed = 42  # None
#trainLen = 168
#testLen =82

# dataset 3:monthly-closings-of-the-dowjones.csv  #todo:  good
#dateparse = lambda x: datetime.datetime.strptime(x, '%Y-%m')  # %Y-%m-%d
#dowjones = pd.read_csv(r'./dataset/monthly-closings-of-the-dowjones.csv', delimiter=',', parse_dates=[0],date_parser=dateparse).values
#data0 = dowjones[:, 1].astype(np.float32)
#data, maxV, minV = normalize(data0,'-01')
# 划分数据集并读取数据
#seed = 42  # None
#trainLen = 189
#testLen =97
#dataset_name = 't3_last'

# dataset 4: monthly-critical-radio-frequenci  #todo:  good
#dateparse = lambda x: datetime.datetime.strptime(x, '%Y-%m')  # %Y-%m-%d
#critical = pd.read_csv(r'./dataset/monthly-critical-radio-frequenci.csv', delimiter=',', parse_dates=[0],date_parser=dateparse).values
#data0 = critical[:, 1].astype(np.float32)
#data, maxV, minV = normalize(data0,'-01')
# 划分数据集并读取数据
#seed = 42  # None
#trainLen = 154
#testLen =79
#dataset_name = 't4_last'

# dataset 5:  co2-ppm-mauna-loa-19651980.csv  #todo:  good
#dateparse = lambda x: datetime.datetime.strptime(x, '%Y-%m')  # %Y-%m-%d
#co2 = pd.read_csv(r'./dataset/co2-ppm-mauna-loa-19651980.csv', delimiter=',', parse_dates=[0],date_parser=dateparse).values
#data0 = co2[:, 1].astype(np.float32)
#data, maxV, minV = normalize(data0,'-01')
# 划分数据集并读取数据
#seed = 42  # None
#trainLen = 122
#testLen =64
#dataset_name = 't5_last'

# dataset 6: monthly-lake-erie-levels-1921-19.csv #todo:  good
#dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m')  # %Y-%m-%d
#lake = pd.read_csv(r'./dataset/monthly-lake-erie-levels-1921-19.csv', delimiter=',', parse_dates=[0],date_parser=dateparse).values
#data0 = lake[:, 1].astype(np.float32)
#data, maxV, minV = normalize(data0,'-01')
# 划分数据集并读取数据
#seed = 42  # None
#trainLen = 390
#testLen =195
#dataset_name = 't6_last'

# dataset 7: BrentOilPrice.csv #todo:  good
BrentOilPrice = pd.read_csv(r'./dataset/BrentOilPrices.csv', delimiter=',').values
data0 = BrentOilPrice[:, 1].astype(np.float32)
data, maxV, minV = normalize(data0, '-01')
# 划分数据集并读取数据
seed = 42  # None
trainLen = 5495
testLen =2712
dataset_name = 't7_last'
# Set hyperparameters
inSize=outSize=1
resSize=56
spectral_radius=0.63
sigma=0.3
input_scaling=2
reg=1e-4
_k=np.arange(2,15,1)
initLen=8
α = 0.2

# dataset 8: sunspot3132.csv #todo:  good
#sunspot = pd.read_csv(r'./dataset/sunspot3132.csv', delimiter=';', header=None).values
#data0 = sunspot[:, 3].astype('float32')
#data, maxV, minV = normalize(data0,'-01')
# 划分数据集并读取数据
#seed = 42  # None
#trainLen = 2092
#testLen =1039
#dataset_name = 't8_last'

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
import numpy as np
book = xlwt.Workbook()
sheet = book.add_sheet('sheet1',cell_overwrite_ok='True')  # 创建sheet页
data1 = []
mode = 'prediction'
min_rmse = math.inf
RMSE_V = 0
MAE_V = 0
MSE_V = 0
RMSE_std = []
MSE_std = []
MAE_std = []

Win = (np.random.rand(resSize, inSize + 1) - 0.5) * input_scaling
W = (np.random.rand(resSize, resSize) - 0.5) * input_scaling
Z = np.random.binomial(1, 1 - sigma, (resSize, resSize))
W = np.multiply(W, Z)  # Element-wise multiply to get desired sparsity按元素相乘以获得所需的稀疏度
rhoW = max(abs(linalg.eig(W)[0]))  # 返回W的特征值，取最大值得到谱半径
W *= spectral_radius / rhoW
H = np.zeros((1 + inSize + resSize, trainLen - initLen))
u = np.zeros((resSize, 1))#1时刻的u0
U = np.zeros((2,resSize))#1时刻的U0，为零矩阵
Yt = data[None, initLen + 1:trainLen + 1]
for k in _k:
    for i in range(30):
        print('k=', k)
        Wstate = (np.random.rand(resSize, k) - 0.5) * input_scaling
        for t in range(trainLen):
            x = data[t]
            # ESNP update equation更新方程
            u = α * u + np.diagonal(np.dot(W, np.tanh(
                np.dot(Win, np.vstack((1, x))) +
                    np.dot(Wstate, U)))).reshape(resSize, 1)
            # print("t>0时u的维度是",u.shape)
            U = np.r_[u.T, U]
            U = U[0:k, :]
            # print("U[k]=", U)
            # print("t>0时U1的维度是", U.shape)
            if t >= initLen:  # ?????
                H[:, t - initLen] = np.vstack((1, x, u))[:, 0]
                # 岭回归求Wout
        H_T = H.T
        if reg is not None:
            # use ridge regression
            Wout = np.dot(np.dot(Yt, H_T),
                          linalg.inv(
                              np.dot(H, H_T) + reg * np.eye(
                                  1 + inSize + resSize)))
        else:
            # 使用伪逆
            Wout = np.dot(Yt, linalg.pinv(H))
        Y = np.zeros((outSize, testLen))
        x = data[trainLen]
        for t in range(testLen):
            u = α * u + np.diagonal(np.dot(W, np.tanh(
                np.dot(Win, np.vstack((1, x))) +
                    np.dot(Wstate, U)))).reshape(resSize, 1)
            U = np.r_[u.T, U]
            U = U[0:k, :]
            y = np.dot(Wout, np.vstack((1, x, u)))
            Y[:, t] = y
            if mode == 'generative':
                x = y
            elif mode == 'prediction':
                x = data[trainLen + t + 1]

        # compute MSE,RMSE,MAE for the first errorLen time steps
        errorLen = testLen
        np.savetxt('./predict/' + dataset_name + str(k) + '.csv', Y[0, 0:errorLen], delimiter=',')
        #Y = re_normalize(Y.reshape((Y.shape[1])), maxV, minV, '-01')
        #mse = sum(np.square(data0[trainLen + 1:trainLen + errorLen + 1] - Y[0:errorLen])) / errorLen
        # the mean square error
        mse = sum(np.square(data[0:trainLen + 1:trainLen + errorLen + 1] - Y[0, 0:errorLen])) / errorLen
        MSE_std.append(mse)
        # residual standard deviation
        rmse = sqrt(mean_squared_error(data[trainLen + 1:trainLen + errorLen + 1], Y[0, 0:errorLen]))
        #rmse = sqrt(mean_squared_error(data0[trainLen + 1:trainLen + errorLen + 1], Y[0:errorLen]))
        RMSE_std.append(rmse)

        #  the mean absolute error
        mae = median_absolute_error(data[trainLen + 1:trainLen + errorLen + 1], Y[0, 0:errorLen])
        #mae = median_absolute_error(data0[trainLen + 1:trainLen + errorLen + 1], Y[0:errorLen])
        MAE_std.append(mae)
        MAE_V += mae
        MSE_V += mse
        RMSE_V += rmse
        print('RMSE=' + str(RMSE_V ) + '±' + str(np.std(RMSE_std)))
        print('MAE=' + str(MAE_V ) + '±' + str(np.std(MAE_std)))
        print('MSE=' + str(MSE_V ) + '±' + str(np.std(MSE_std)))

    U = np.r_[u.T, U]
    U = U[0:k + 1, :]

    print('RMSE=' + str(RMSE_V/30 ) + '±' + str(np.std(RMSE_std)))
    print('MAE=' + str(MAE_V/30 ) + '±' + str(np.std(MAE_std)))
    print('MSE=' + str(MSE_V/30) + '±' + str(np.std(MSE_std)))
    RMSE = str(RMSE_V/30)
    MAE = str(MAE_V/30)
    MSE = str(MSE_V/30)
    if (RMSE_V/30) < min_rmse:
        min_rmse = RMSE_V/30
        resSize_N = resSize
    print('resSize_N=' + str(resSize_N))
    data1.append(k)
    data1.append(RMSE)
    data1.append(MAE)
    data1.append(MSE)
    Y = re_normalize(Y.reshape((Y.shape[1])), maxV, minV, '-01')
    fig4 = plt.figure()
    ax41 = fig4.add_subplot(111)
    time = range(testLen)
    ax41.plot(time, data0[trainLen + 1:trainLen + testLen + 1], 'r-', label='the original data')
    ax41.plot(time, Y[0:errorLen], 'g--', label='the predicted data')
    ax41.set_ylabel("Magnitude")
    ax41.set_xlabel('Time')
    ax41.set_title('1BrentoilPrices')
    plt.savefig('./picture/' + 'BrentoilPrices' + str(k) + '.png')
    fig5 = plt.figure()
    ax5 = fig5.add_subplot(111)
    ax5.plot(time, np.abs(data0[trainLen + 1:trainLen + testLen + 1] - Y[0:errorLen]), 'r')
    ax5.set_ylabel("Magnitude")
    ax5.set_xlabel('Time')
    ax5.set_title('BrentoilPrices')
    plt.savefig('./picture/' + '1BrentoilPrices' + str(k) + '.png')
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

data1 = list_of_groups(data1, 4)
print("data1=", data1)
import pandas as pd
df=pd.DataFrame(data1,columns=['k','RMSE','MAE','MSE'])
df.to_excel('./result1/MG结果.xlsx')

