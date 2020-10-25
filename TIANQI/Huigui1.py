import numpy as np
from numpy.core.arrayprint import dtype_is_implied
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
highs = data['最高气温']
highs = highs.astype(int)
# 日次操作同理，这里不再赘述
data['日期'] = data['日期'].map(lambda x:x.replace('2020年10月',''))
data['日期'] = data['日期'].map(lambda x:x.replace('日',''))
datas = data['日期']
# 不理解的小伙伴可运行下两行代码查看运行结果（这里先注释掉了）
# print(data['日期'])
# print(data['最高气温'])
 
# 传入对应日期及其最高气温参数
# # 应以矩阵形式表达(对于单变量，矩阵就是列向量形式)
xTrain = np.array(datas, dtype=np.int)[:, np.newaxis]
# 为方便理解，也转换成列向量
yTrain = np.array(highs,dtype=np.int)
# yTrain = np.array([18,19,20,22,24,25,26])
 
# 创建模型对象
model = LinearRegression()
# 根据训练数据拟合出直线(以得到假设函数)
hypothesis = model.fit(xTrain, yTrain)
# 截距
print("theta0=", hypothesis.intercept_)
# 斜率
print("theta1=", hypothesis.coef_)
 
# 预测2020年9月12日的最高气温
print("预测2020年10月12日的最高气温：", model.predict([[12]]))
# 也可以批量预测多个日期的气温，注意要以列向量形式表达（有余数据集量少，故间隔时间长气温可能有较大差异）
# 此处仅利用模型表示，不代表真实值（假设要预测10号、11号、12号的天气）
xNew = np.array([0, 8, 9, 10])[:, np.newaxis]
yNew = model.predict(xNew)
print("预测新数据：", xNew)
print("预测结果：", yNew)
 
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
 
plt = initPlot()  # 画图
# k是黑色，.是以点作为图上显示
plt.plot(xTrain, yTrain, 'k.')
# 画出通过这些点的连续直线
plt.plot(xNew, yNew, 'g--')
# 将图显示出来
plt.show()