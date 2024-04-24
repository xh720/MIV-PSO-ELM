import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

plt.rcParams["font.sans-serif"] = "SimHei"  # 显示中文 黑体中文
plt.rcParams["axes.unicode_minus"] = False  # 显示负号

# ELM 类定义
class ELM:
    def __init__(self, n_hidden, activation_func='sigmoid'):
        self.n_hidden = n_hidden
        self.activation_func = activation_func
        self.input_weights = None
        self.biases = None
        self.output_weights = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def relu(self, x):
        return np.maximum(0, x)

    def activation(self, x):
        if self.activation_func == 'sigmoid':
            return self.sigmoid(x)
        elif self.activation_func == 'tanh':
            return self.tanh(x)
        elif self.activation_func == 'relu':
            return self.relu(x)
        else:
            raise ValueError("Unknown activation function.")

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.input_weights = np.random.normal(0, 1, (n_features, self.n_hidden))
        self.biases = np.random.normal(0, 1, (1, self.n_hidden))
        H = self.activation(np.dot(X, self.input_weights) + self.biases)
        pseudo_inverse_H = np.linalg.pinv(np.dot(H.T, H))
        self.output_weights = np.dot(pseudo_inverse_H, np.dot(H.T, y.reshape(-1, 1)))

    def predict(self, X):
        H = self.activation(np.dot(X, self.input_weights) + self.biases)
        return np.dot(H, self.output_weights).flatten()

    # MIV 计算函数


def calculate_miv(model, X, y):
    n_features = X.shape[1]
    mivs = np.zeros(n_features)

    # 计算每个特征对输出的影响
    for i in range(n_features):
        X_copy = X.copy()
        X_copy[:, i] = np.random.normal(0, 1, X_copy.shape[0])  # 打乱当前特征
        y_pred_orig = model.predict(X)
        y_pred_pert = model.predict(X_copy)
        mivs[i] = np.mean((y_pred_orig - y_pred_pert) ** 2)  # 计算均方误差

    return mivs


# 示例
if __name__ == "__main__":
    # 生成模拟数据
    # 加载数据
    data = pd.read_excel('F:\Python\python_program\pythonProject\A\数据集.xlsx')

    # 标准化处理
    scaler = StandardScaler()
    X = data.iloc[1:, :-1]  # iloc表示按位置取值， :代表取所有行， :-1表示取除了最后一列的所有列
    # 将数据标准化处理
    X = scaler.fit_transform(X)
    # 输出值不作处理
    y = data.iloc[1:, -1]  # -1表示取最后一列
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_train = y_train.values.reshape(-1, 1)
    print(type(y_train))
    # 初始化 ELM 模型
    elm = ELM(n_hidden=50, activation_func='sigmoid')

    # 训练 ELM 模型
    elm.fit(X_train, y_train)

    # 预测
    y_pred = elm.predict(X_test)

    # 计算并打印 MIV
    mivs = calculate_miv(elm, X_train, y_train)
    print("MIVs:", mivs)

    # 你可以选择对 MIVs 进行排序，找出最重要的变量
    sorted_mivs = np.argsort(mivs)[::-1]
    print("Most influential variables:", sorted_mivs)

    # 根据 MIV 对特征进行排序
    sorted_indices = np.argsort(mivs)[::-1]

    # 选择前两个最具影响力的特征
    selected_indices = sorted_indices[:2]

    # 选择相应的特征
    X_train_selected = X_train[:, selected_indices]
    X_test_selected = X_test[:, selected_indices]

    # 初始化 ELM 模型
    elm_selected = ELM(n_hidden=50, activation_func='sigmoid')

    # 使用选定的特征训练 ELM 模型
    elm_selected.fit(X_train_selected, y_train)

    # 使用选定的特征进行预测
    y_pred_selected = elm_selected.predict(X_test_selected)

    # 评估模型性能
    mse_selected = mean_squared_error(y_test, y_pred_selected)
    print(f"Mean Squared Error using {len(selected_indices)} most influential variables: {mse_selected}")


    # 打印所选特征
    print("Selected features:", selected_indices + 1)  # 转换为原始特征的索引（因为Python的索引从0开始，但通常特征编号从1开始）

# 评估模型性能
mse = mean_squared_error(y_test, y_pred_selected)
print(f"Mean Squared Error: {mse}")

rmse = np.sqrt(np.mean((y_test - y_pred_selected) ** 2))
print(f"RMSE:{rmse}")

r2 = r2_score(y_test, y_pred_selected)
print(f"R2:{r2}")

# 你还可以绘制 MIVs 的条形图
import matplotlib.pyplot as plt

plt.figure(1)
plt.bar(range(len(mivs)), mivs)
plt.xlabel('Variables')
plt.ylabel('MIV')
plt.title('Most Influential Variables')
plt.xticks(range(len(mivs)), range(1, len(mivs) + 1))
plt.show()


plt.figure(2)
plt.title('测试集预测结果对比')
plt.plot(range(len(y_pred_selected)), y_pred_selected, 'b-o', linewidth=1, label='预测值')
plt.plot(range(len(y_test)), y_test, 'r-*', linewidth=1, label='真实值')
plt.legend()
plt.xlabel('预测样本', size=14)
plt.ylabel('预测结果', size=14)
plt.show()