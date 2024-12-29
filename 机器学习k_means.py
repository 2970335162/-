import pandas as pd
import numpy as np
from sklearn.cluster import KMeans  # 使用KMeans聚类算法
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix  # 导入混淆矩阵
import seaborn as sns  # 导入 seaborn，用于可视化
# 定义文件路径和列名
column_names = [f'feature_{i+1}' for i in range(52)] + ['fault_label']  # 52个特征 + 目标列

# 加载数据的函数
def load_data(file_path, fault_label=None):
    # 加载数据，假设每个数据文件是以空格分隔的
    data = pd.read_csv(file_path, delim_whitespace=True, header=None, names=column_names[:-1])
    # 如果有故障标签，添加到数据中
    if fault_label is not None:
        data['fault_label'] = fault_label
    else:
        data['fault_label'] = 0  # 正常数据标签为0
    return data

# 训练集和测试集的文件路径
train_files = [f'C:\\Users\\关明珠\\Desktop\\te数据集\\训练集\\d{str(i).zfill(2)}.dat' for i in range(22)]
test_files = [f'C:\\Users\\关明珠\\Desktop\\te数据集\\测试集\\d{str(i).zfill(2)}_te.dat' for i in range(22)]

# 加载训练集数据
train_data_list = []
for i, file in enumerate(train_files):
    fault_label = 0 if i == 0 else i  # d00.dat为正常数据，其他为故障数据
    train_data_list.append(load_data(file, fault_label))

train_data = pd.concat(train_data_list, ignore_index=True)

# 加载测试集数据
test_data_list = []
for i, file in enumerate(test_files):
    fault_label = 0 if i == 0 else i  # d00_te.dat为正常数据，其他为故障数据
    test_data_list.append(load_data(file, fault_label))

test_data = pd.concat(test_data_list, ignore_index=True)

# 查看数据的基本信息
print("训练集基本信息:")
print(train_data.head())
print(train_data.info())

print("测试集基本信息:")
print(test_data.head())
print(test_data.info())

# 特征列和目标列
X_train = train_data.drop(columns=['fault_label'])
y_train = train_data['fault_label']

X_test = test_data.drop(columns=['fault_label'])
y_test = test_data['fault_label']

# 数据标准化，KMeans对特征的尺度敏感
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # 使用训练集的scaler对测试集进行转换

# 创建KMeans聚类模型
kmeans = KMeans(n_clusters=2, random_state=42)  # 假设数据分为2类：正常和故障
kmeans.fit(X_train_scaled)

# 使用KMeans模型预测测试集的簇标签
y_pred = kmeans.predict(X_test_scaled)

# 将聚类标签映射为正常和故障标签（假设标签0为正常，标签1为故障）
# 如果KMeans预测结果的簇0代表故障，簇1代表正常，则需要进行标签交换
if np.sum(y_pred == 0) > np.sum(y_pred == 1):  # 如果大部分测试数据被分配到标签0，则认为标签0是正常
    y_pred = 1 - y_pred  # 交换标签

# 打印分类报告
print("分类报告:\n", classification_report(y_test, y_pred))

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"准确率: {accuracy:.4f}")

# 计算混淆矩阵
cm = confusion_matrix(y_test, y_pred)

# 可视化混淆矩阵
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['normal', 'error'], yticklabels=['normal', 'error'])
plt.title('Confusion Matrix')
plt.xlabel('Prediction Label')
plt.ylabel('True Label')
plt.show()