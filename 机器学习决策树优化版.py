import pandas as pd  # 导入 pandas 为 PD
import numpy as np  # 导入 numpy 为 NP
from sklearn.tree import DecisionTreeClassifier  # 导入决策树
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix  # 导入评估方法
from sklearn.preprocessing import StandardScaler  # 导入标准化方法
from sklearn.model_selection import train_test_split  # 导入训练集和测试集拆分方法
from sklearn.feature_selection import SelectFromModel  # 导入特征选择方法
import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 为 PLT
import seaborn as sns  # 导入 Seaborn 为 SNS

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
X_test_scaled = scaler.transform(X_test)  # 使用训练集的scaler对测试集进行转换

# 创建决策树模型
dt_model = DecisionTreeClassifier()

# 特征选择：基于决策树的特征重要性
sfm = SelectFromModel(dt_model, threshold="mean")  # 选择平均重要性的特征
sfm.fit(X_train_scaled, y_train)

# 提取选择的特征
X_train_selected = sfm.transform(X_train_scaled)
X_test_selected = sfm.transform(X_test_scaled)

# 训练模型
dt_model.fit(X_train_selected, y_train)

# 预测
y_pred = dt_model.predict(X_test_selected)

# 打印分类报告
print("分类报告：\n", classification_report(y_test, y_pred))

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"准确率：{accuracy:.4f}")

# 计算混淆矩阵
cm = confusion_matrix(y_test, y_pred)

# 可视化混淆矩阵
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['normal', 'error'], yticklabels=['normal', 'error'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()


# 训练过程中准确率变化曲线
# 拆分训练集为训练集和验证集
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train_selected, y_train, test_size=0.2, random_state=42)

# 初始化列表来记录训练过程中的准确率变化
train_accuracies = []
val_accuracies = []

# 逐步训练并记录准确率
for i in range(1, len(X_train_split) + 1, 100):  # 每100个样本训练一次
    dt_model.fit(X_train_split[:i], y_train_split[:i])  # 使用逐步增加样本的方式训练
    train_accuracy = accuracy_score(y_train_split[:i], dt_model.predict(X_train_split[:i]))
    val_accuracy = accuracy_score(y_val_split, dt_model.predict(X_val_split))  # 在验证集上评估准确率
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)

# 可视化训练准确率和验证准确率变化曲线
plt.figure(figsize=(10, 6))
plt.plot(train_accuracies, label='Training Accuracy', color='blue')
plt.plot(val_accuracies, label='Validation Accuracy', color='orange')
plt.xlabel('Training Steps (Samples)')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy Over Time')
plt.legend()
plt.grid(True)
plt.show()
