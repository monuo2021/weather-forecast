import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 解决中文问题（若没有此步骤，表名字及横纵坐标中的汉语将无法显示[具体会显示矩形小方格]）
plt.rcParams['font.sans-serif'] = ['SimHei']

# 将数据从上一步存入的 .csv 格式文件中读取
data = pd.read_csv(r'D:/Users/文档/program/Python/TIANQI3/NanChang10Mouth3.csv')
# 由于最高气温与最低气温中有 / 分隔，故将其分开，即“气温”列由一列变为两列——“最高气温”和“最低气温”
data['最高气温'] = data['气温'].str.split('/',expand=True)[0]
# 我们要对数值进行分析，所以将多余的单位 ℃ 从列表中去掉，只保留数值部分
data['最高气温'] = data['最高气温'].map(lambda x:x.replace('℃,',''))
highs = data['最高气温']
highs = highs.astype(int)

# data['最低气温'] = data['气温'].str.split('/,',expand=True)[1]
# data['最低气温'] = data['最低气温'].map(lambda x:x.replace('℃',''))
# lows = data['最低气温']
# lows = lows.astype(int)
# 日次操作同理，这里不再赘述
data['日期'] = data['日期'].map(lambda x:x.replace('2020年10月',''))
data['日期'] = data['日期'].map(lambda x:x.replace('日',''))
datas = data['日期']
# 不理解的小伙伴可运行下两行代码查看运行结果（这里先注释掉了）
# print(data['日期'])
# print(data['最高气温'])

def initPlot():
    # 先准备好一块画布
    plt.figure()
    # 生成图表的名字
    plt.title('2020年10月上旬南昌天气')
    # 横坐标名字
    plt.xlabel('日期')
    # 纵坐标名字
    plt.ylabel('当日最高气温/℃')
    # 表内有栅格（不想要栅格把此行注释掉即可）
    plt.grid(True) 
    return plt

plt = initPlot()  # 画图
# 传入对应日期及其最高气温参数
xTrain = np.array(datas[4:],dtype=np.int)
yTrain = np.array(highs[4:])
# k是黑色，.是以点作为图上显示
plt.plot(xTrain, yTrain, 'k.')
# 将图显示出来
plt.show()