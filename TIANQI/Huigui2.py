import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# 解决中文问题（若没有此步骤，表名字及横纵坐标中的汉语将无法显示[具体会显示矩形小方格]）
plt.rcParams['font.sans-serif'] = ['SimHei']
 
# 将数据从上一步存入的 .csv 格式文件中读取
data = pd.read_csv(r'D:/Users/文档/编程/Python/TIANQI/NanChang10Mouth.csv')
# 由于最高气温与最低气温中有 / 分隔，故将其分开，即“气温”列由一列变为两列——“最高气温”和“最低气温”
data['最高气温'] = data['气温'].str.split('/',expand=True)[0]
# 我们要对数值进行分析，所以将多余的单位 ℃ 从列表中去掉，只保留数值部分
data['最高气温'] = data['最高气温'].map(lambda x:x.replace('℃,',''))
# 日次操作同理，这里不再赘述
data['日期'] = data['日期'].map(lambda x:x.replace('2020年05月0',''))
data['日期'] = data['日期'].map(lambda x:x.replace('日',''))
# 不理解的小伙伴可运行下两行代码查看运行结果（这里先注释掉了）
# print(data['日期'])
# print(data['最高气温'])
 
# 传入对应日期及其最高气温参数
# # # 应以矩阵形式表达(对于单变量，矩阵就是列向量形式)
# xTrain = np.array(data['日期'])[:, np.newaxis]
# # 为方便理解，也转换成列向量
# yTrain = np.array(data['最高气温'])
xTrain = np.array([1,2,3,4,5,6,7])[:, np.newaxis]  # 训练数据（日期）
yTrain = np.array([18,19,20,22,24,25,26])        # 训练数据（最高气温）
xTest = np.array([2,3,5,7])[:,np.newaxis]          # 测试数据（日期）
yTest = np.array([19,20,24,26])                    # 测试数据（最高气温）
# 创建模型对象
model = LinearRegression()
# 根据训练数据拟合出直线(以得到假设函数)
hypothesis = model.fit(xTrain, yTrain)
hpyTrain = model.predict(xTrain)
# 针对测试数据进行预测
hpyTest = model.predict(xTest)
 
# 手动计算训练数据集残差
ssResTrain = sum((hpyTrain - yTrain) ** 2)
print(ssResTrain)
# Python计算的训练数据集残差
print(model._residues)
 
# 手动计算测试数据集残差
ssResTest = sum((hpyTest - yTest) ** 2)
# 手动计算测试数据集y值偏差平方和
ssTotTest = sum((yTest - np.mean(yTest)) ** 2)
# 手动计算R方
Rsquare = 1 - ssResTest / ssTotTest
print(Rsquare)
# Python计算的训练数据集的R方
print(model.score(xTest, yTest))
 
# corrcoef函数是在各行元素之间计算相关性，所以x和y都应是行向量
print(np.corrcoef(xTrain.T, yTrain.T))  # 计算训练数据的相关性
print(np.corrcoef(xTest.T, yTest.T))    # 计算测试数据的相关性
 
def initPlot():
    # 先准备好一块画布
    plt.figure()
    # 生成图表的名字
    plt.title('2020年10月上旬南昌天气')
    # 横坐标名字
    plt.xlabel('日期')
    # 纵坐标名字
    plt.ylabel('当日最高气温')
    # 表内有栅格（不想要栅格把此行注释掉即可）
    plt.grid(True)
    return plt
 
plt = initPlot()
plt.plot(xTrain, yTrain, 'r.')          # 训练点数据(红色)
plt.plot(xTest, yTest, 'b.')            # 测试点数据(蓝色)
plt.plot(xTrain, hpyTrain, 'g-')        # 假设函数直线(绿色)
plt.show()