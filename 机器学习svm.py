import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV  # 导入 GridSearchCV
import timeit

# 定义文件路径和列名
column_names = [f'feature_{i+1}' for i in range(52)] + ['fault_label']  # 52个特征 + 目标列

# 加载数据的函数
def load_data(file_path, fault_label=None):
    data = pd.read_csv(file_path, delim_whitespace=True, header=None, names=column_names[:-1])
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
    fault_label = i if i != 0 else 0  # d00.dat为正常数据，其他为故障数据
    train_data_list.append(load_data(file, fault_label))

train_data = pd.concat(train_data_list, ignore_index=True)

# 加载测试集数据
test_data_list = []
for i, file in enumerate(test_files):
    fault_label = i if i != 0 else 0  # d00_te.dat为正常数据，其他为故障数据
    test_data_list.append(load_data(file, fault_label))

test_data = pd.concat(test_data_list, ignore_index=True)

# 查看数据的基本信息
print("训练集基本信息：")
print(train_data.head())
print(train_data.info())

print("测试集基本信息：")
print(test_data.head())
print(test_data.info())

# 特征列和目标列
X_train = train_data.drop(columns=['fault_label'])
y_train = train_data['fault_label']

X_test = test_data.drop(columns=['fault_label'])
y_test = test_data['fault_label']

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 创建SVM模型
svm_model = SVC(kernel='linear')  # 使用线性核SVM

# 使用 GridSearchCV 进行交叉验证并记录训练过程中的准确率
param_grid = {'C': [0.1, 1, 10, 100]}  # 可以调整C参数的取值范围
grid_search = GridSearchCV(svm_model, param_grid, cv=5, verbose=1, return_train_score=True)

# 训练模型并记录准确率
grid_search.fit(X_train_scaled, y_train)

# 打印最佳参数
print("最佳参数：", grid_search.best_params_)

# 绘制训练准确率和验证准确率的变化曲线
train_scores = grid_search.cv_results_['mean_train_score']
val_scores = grid_search.cv_results_['mean_test_score']

plt.figure(figsize=(8, 6))
plt.plot(param_grid['C'], train_scores, label='Train Accuracy', marker='o')
plt.plot(param_grid['C'], val_scores, label='Validation Accuracy', marker='o')
plt.xscale('log')
plt.xlabel('C Parameter')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy vs C parameter')
plt.legend()
plt.grid(True)
plt.show()

# 评估测试集性能
y_pred = grid_search.best_estimator_.predict(X_test_scaled)

# 打印分类报告
print("分类报告：\n", classification_report(y_test, y_pred))

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"准确率： {accuracy:.4f}")

# 计算混淆矩阵
cm = confusion_matrix(y_test, y_pred)

# 可视化混淆矩阵
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['normal', 'error'], yticklabels=['normal', 'error'])
plt.title('Confusion Matrix')
plt.xlabel('Prediction Label')
plt.ylabel('True Label')
plt.show()





