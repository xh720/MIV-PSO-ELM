import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from mivpso import PSO_MIV
from mivpso import ELM_MIV

plt.rcParams["font.sans-serif"] = "SimHei"  # 显示中文 黑体中文
plt.rcParams["axes.unicode_minus"] = False  # 显示负号

# 极限学习机类
class ELM:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weights_input_to_hidden = np.random.rand(input_size, hidden_size)
        self.bias_hidden = np.random.rand(1, hidden_size)
        self.weights_hidden_to_output = None

    def fit(self, X, Y, weights_input_to_hidden=None, bias_hidden=None):
        if weights_input_to_hidden is not None:
            self.weights_input_to_hidden = weights_input_to_hidden
        if bias_hidden is not None:
            self.bias_hidden = bias_hidden
        H = np.dot(X, self.weights_input_to_hidden) + self.bias_hidden
        H = 1 / (1 + np.exp(-H))  # 使用sigmoid激活函数
        self.weights_hidden_to_output = np.dot(np.linalg.pinv(H), Y)

    def predict(self, X):
        H = np.dot(X, self.weights_input_to_hidden) + self.bias_hidden
        H = 1 / (1 + np.exp(-H))
        Y_pred = np.dot(H, self.weights_hidden_to_output)
        return Y_pred

    def get_weights(self):
        return self.weights_input_to_hidden
            # , self.weights_hidden_to_output

    def get_biases(self):
        return self.bias_hidden

    def set_weights_and_biases(self, weights_input_to_hidden, bias_hidden):
        self.weights_input_to_hidden = weights_input_to_hidden
        # weights_hidden_to_output,
        # self.weights_hidden_to_output = weights_hidden_to_output
        self.bias_hidden = bias_hidden


    # 粒子群优化类


class PSO:
    def __init__(self, num_particles, num_iterations, dim, omega=0.5, c1=1.45, c2=1.65):
        # 设置粒子群中的粒子数量
        self.num_particles = num_particles
        # 设置优化算法的最大迭代次数。
        self.num_iterations = num_iterations
        # 设置问题的维度，即每个粒子在搜索空间中的位置向量的长度。
        self.dim = dim
        # 设置惯性权重，用于控制粒子速度更新的程度。
        self.omega = omega
        # 设置认知系数，影响粒子向自身历史最佳位置移动的程度。
        self.c1 = c1
        # 设置社会系数，影响粒子向群体最佳位置移动的程度。
        self.c2 = c2
        # 初始化粒子群的位置。np.random.rand生成一个形状为(num_particles, dim)的数组，其中的元素是从均匀分布中随机抽取的，范围在0到1之间。
        self.particles = np.random.rand(num_particles, dim)
        # 初始化粒子群的速度
        self.velocities = np.random.rand(num_particles, dim)
        # 初始化每个粒子的历史最佳位置。开始时，每个粒子的历史最佳位置就是其当前位置。
        self.best_positions = np.copy(self.particles)
        # 初始化每个粒子的历史最佳分数（适应度值）。开始时，所有粒子的最佳分数被设置为无穷大（np.inf），表示还没有找到任何好的解。
        self.best_scores = np.full(num_particles, np.inf)
        # 初始化全局最佳位置。开始时，没有全局最佳位置，所以设置为None。
        self.global_best_position = None
        # 初始化全局最佳分数。开始时，全局最佳分数被设置为无穷大，表示还没有找到全局最优解。
        self.global_best_score = np.inf

    def evaluate(self, elm, X_train, y_train, X_val, y_val):
        # 从粒子位置数组的第一行（通常代表第一个粒子的位置）取出前`elm.input_size * elm.hidden_size`个元素。
        # 这些元素对应于ELM输入层到隐藏层的权重。权重矩阵的每一行对应一个输入节点，每一列对应一个隐藏节点。
        elm.weights_input_to_hidden = self.particles[0, :elm.input_size * elm.hidden_size].reshape(elm.input_size,
                                                                                                   elm.hidden_size)
        # 从粒子位置数组的第一行取出从 `elm.input_size * elm.hidden_size` 开始到最后的所有元素。这些元素对应于ELM隐藏层的偏置。
        # 偏置是一个行向量，每个元素对应一个隐藏节点。
        elm.bias_hidden = self.particles[0, elm.input_size * elm.hidden_size:].reshape(1, elm.hidden_size)
        elm.fit(X_train, y_train)
        y_pred = elm.predict(X_val)
        score = mean_squared_error(y_val, y_pred)
        return score

    # 指定要更新的粒子的索引
    def update_velocity_and_position(self, index):
        # 生成一个与问题维度相同长度的随机向量r1，其中的元素在0到1之间均匀分布。这个随机向量用于认知部分的更新。
        r1 = np.random.rand(self.dim)
        # 社会部分更新
        r2 = np.random.rand(self.dim)
        velocity_cognitive = self.c1 * r1 * (self.best_positions[index] - self.particles[index])
        velocity_social = self.c2 * r2 * (self.global_best_position - self.particles[index])
        self.velocities[index] = self.omega * self.velocities[index] + velocity_cognitive + velocity_social
        # 根据更新后的速度，更新粒子的位置
        self.particles[index] += self.velocities[index]

    def optimize(self, elm, X_train, y_train, X_test, y_test):
        self.best_scores_history = []  # 用于记录每一代的最佳适应度值
        for _ in range(self.num_iterations):
            for i in range(self.num_particles):
                score = self.evaluate(elm, X_train, y_train, X_test, y_test)
                if score < self.best_scores[i]:
                    self.best_scores[i] = score
                    self.best_positions[i] = np.copy(self.particles[i])
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = np.copy(self.particles[i])
            for i in range(self.num_particles):
                self.update_velocity_and_position(i)
                # 计算并记录当前代的最佳适应度值
            best_score_this_iteration = min(self.best_scores)
            self.best_scores_history.append(best_score_this_iteration)
        return self.global_best_position
        # elm.weights_input_to_hidden = self.global_best_position[:elm.input_size * elm.hidden_size].reshape(
        #     elm.input_size, elm.hidden_size)
        # elm.bias_hidden = self.global_best_position[elm.input_size * elm.hidden_size:].reshape(1, elm.hidden_size)

    def get_weights(self):
        if self.global_best_position is None:
            raise ValueError("No global best position found. Please run optimization first.")
        weights_input_to_hidden = self.global_best_position[:self.elm_input_size * self.elm_hidden_size].reshape(
            self.elm_input_size, self.elm_hidden_size)
        return weights_input_to_hidden

    def get_biases(self):
        if self.global_best_position is None:
            raise ValueError("No global best position found. Please run optimization first.")
        biases_hidden = self.global_best_position[self.elm_input_size * self.elm_hidden_size:].reshape(1,
                                                                                                       self.elm_hidden_size)
        return biases_hidden

    def set_weights_and_biases(self, elm):
        weights = self.get_weights()
        biases = self.get_biases()
        elm.weights_input_to_hidden = weights
        elm.bias_hidden = biases

    def plot_optimization_process(self):
        plt.plot(self.best_scores_history, label='迭代中最佳适应度值')
        plt.xlabel('迭代次数')
        plt.ylabel('适应度值MSE')
        plt.title('PSO误差迭代过程')
        plt.legend()
        plt.show()

def calculate_miv(model, X, y, perturbation_ratio=0.1):
    n_features = X.shape[1]
    mivs = np.zeros(n_features)

    # 计算每个特征对输出的影响
    for i in range(n_features):
        # 对第i个特征进行正向和负向扰动
        X_pos_pert = X.copy()
        # print(X_pos_pert.shape)
        X_pos_pert[:, i] += perturbation_ratio * X_pos_pert[:, 1]  # 打乱当前特征
        X_neg_pert = X.copy()
        X_neg_pert[:, i] -= perturbation_ratio * X_neg_pert[:, 1]

        # 获取原始预测、正向扰动后的预测和负向扰动后的预测
        y_pred_orig = model.predict(X)
        y_pred_pos_pert = model.predict(X_pos_pert)
        y_pred_neg_pert = model.predict(X_neg_pert)

        # 计算正向和负向影响变化值（IV）
        iv_pos = np.mean((y_pred_orig - y_pred_pos_pert) ** 2)
        iv_neg = np.mean((y_pred_orig - y_pred_neg_pert) ** 2)

        # 取平均值作为该特征的MIV值
        mivs[i] = (iv_pos + iv_neg) / 2  # 计算均方误差

    return mivs

    data = pd.read_excel('dbn.xlsx')
    input_data = data.iloc[:, :-1].values
    output_data = data.iloc[:, -1].values.reshape(-1, 1)
    # 归一化
    # ss_X = StandardScaler().fit(input_data)
    # ss_y = StandardScaler().fit(output_data)
    ss_X = MinMaxScaler(feature_range=(0, 1)).fit(input_data)
    ss_y = MinMaxScaler(feature_range=(0, 1)).fit(output_data)
    input_data = ss_X.transform(input_data)
    output_data = ss_y.transform(output_data)

    n = input_data.shape[0]
    m = n - 100
    # 划分训练集和测试集
    # 注意Python的切片操作是左闭右开的
    X_train = input_data[:m, :]  # 训练集数据
    X_test = input_data[m:, :]  # 测试集数据
    y_train = output_data[:m, :]  # 训练集标签
    y_test = output_data[m:, :]  # 测试集标签

    # 加载数据
data = pd.read_excel('F:\Python\python_program\pythonProject\A\数据集.xlsx')

# 标准化处理
# scaler_X = MinMaxScaler(feature_range=(0, 1))
# scaler_y = MinMaxScaler(feature_range=(0, 1))
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = data.iloc[1:, :-1]  # iloc表示按位置取值， :代表取所有行， :-1表示取除了最后一列的所有列
y = data.iloc[1:, -1]  # -1表示取最后一列
# 将数据归一化标准化处理
# X = scaler_X.fit_transform(X)
# y = scaler_y.fit_transform(y.values.reshape(-1, 1))
# 随机划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

# scaler_y = StandardScaler()

# y_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
# y_test = scaler_y.transform(y_test.values.reshape(-1, 1))

y_train = y_train.values.reshape(-1, 1)
y_test = y_test.values.reshape(-1, 1)
# print("训练集的特征值：\n", X_train, X_train.shape)

# # 生成模拟数据
# X, Y = make_regression(n_samples=100, n_features=10, n_informative=2, noise=0.1)
# X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# 初始化ELM和PSO
input_size = X_train.shape[1]
hidden_size = 50
output_size = y_train.shape[1]
elm = ELM(input_size, hidden_size, output_size)
num_particles = 40
num_iterations = 100
dim = input_size * hidden_size + hidden_size  # 权重和阈值的总维度
pso = PSO(num_particles, num_iterations, dim)


# 优化ELM的权重和阈值
pso.optimize(elm, X_train, y_train, X_test, y_test)


# 绘制PSO迭代过程图
pso.plot_optimization_process()

# 使用优化后的权重和阈值进行预测
y_pred_p = elm.predict(X_test)

# ptest_pred = scaler_y.inverse_transform(y_pred)
# y_true = scaler_y.inverse_transform(y_test)
y_pred_p = y_pred_p.ravel()
y_true = y_test.ravel()

# 计算粒子群优化结果相关参数
mse = mean_squared_error(y_true, y_pred_p)
print(f"Optimized MSE: {mse}")
# y_pred = y_pred.ravel()
rmse = np.sqrt(np.mean((y_true - y_pred_p) ** 2))
print(f"RMSE:{rmse}")

R2 = r2_score(y_true, y_pred_p)
print(f"R2:{R2}")

MAE = mean_absolute_error(y_true, y_pred_p)
print(f"MAE:{MAE}")

plt.figure()
plt.title('pso测试集预测结果对比')
plt.plot(range(len(y_pred_p)), y_pred_p, 'b-o', linewidth=1, label='预测值')
plt.plot(range(len(y_true)), y_true, 'r-*', linewidth=1, label='真实值')
plt.legend()
plt.xlabel('预测样本', size=14)
plt.ylabel('预测结果', size=14)
plt.show()

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

# # 选择相应的特征
# X_train_selected = X_train[:, selected_indices]
# X_val_selected = X_val[:, selected_indices]

# 打印所选特征
print("Selected features:", selected_indices + 1)  # 转换为原始特征的索引（因为Python的索引从0开始，但通常特征编号从1开始）

plt.figure()
plt.bar(range(len(mivs)), mivs)
plt.xlabel('Variables')
plt.ylabel('MIV')
plt.title('特征变量重要度')
plt.xticks(range(len(mivs)), range(1, len(mivs) + 1))
plt.show()

# 使用选定的特征重新训练ELM模型
X_train_selected = X_train[:, selected_indices]
X_test_selected = X_test[:, selected_indices]
input_size_sel = X_train_selected.shape[1]
elm_miv = ELM_MIV(input_size_sel, hidden_size, output_size)
dims = input_size_sel * hidden_size + hidden_size
# elm_selected.set_weights_and_biases(elm.get_weights(), elm.get_biases())  # 设置优化后的权重和阈值
# elm_selected.fit(X_train_selected, y_train)

# # 使用选定的特征进行预测
# y_pred_selected = elm_reinitialized.predict(X_val_selected)

# 使用筛选后的特征重新初始化ELM和PSO
# elm_reinitialized = ELM(input_size=len(selected_indices), hidden_size=hidden_size, output_size=output_size)
pso_m = PSO_MIV(num_particles, num_iterations, dims)

# 使用筛选后的特征重新进行PSO-ELM训练
pso_m.optimize_M(elm_miv, X_train_selected, y_train, X_test_selected, y_test)
#
# 绘制PSO迭代过程图
pso_m.plot_optimization_process_M()

# 使用重新训练后的PSO-ELM进行预测
y_pred_miv = elm_miv.predict_M(X_test_selected)

# y_pred_miv = scaler_y.inverse_transform(y_pred_retrained)
y_pred_miv = y_pred_miv.ravel()


# input_size_sel = X_train_selected.shape[1]
# # 初始化 ELM 模型
# elm_selected = ELM(input_size_sel, hidden_size, output_size)
#
# # 使用选定的特征训练 ELM 模型
# elm_selected.fit(X_train_selected, y_train)
#
# # 使用选定的特征进行预测
# y_pred_selected = elm_selected.predict(X_val_selected)




# MIV
# y_pred_selected = y_pred_retrained.ravel()
mse_MIV = mean_squared_error(y_true, y_pred_miv)
print(f"Optimized MSE_miv: {mse_MIV}")
# y_pred = y_pred.ravel()
rmse_MIV = np.sqrt(np.mean((y_true - y_pred_miv) ** 2))
print(f"RMSE_miv:{rmse_MIV}")

R2_MIV = r2_score(y_true, y_pred_miv)
print(f"R2_miv:{R2_MIV}")

MAE_MIV = mean_absolute_error(y_true, y_pred_p)
print(f"MAE_miv:{MAE_MIV}")



plt.figure()
plt.title('MIV_pso测试集预测结果对比')
plt.plot(range(len(y_pred_miv)), y_pred_miv, 'b-o', linewidth=1, label='预测值')
plt.plot(range(len(y_true)), y_true, 'r-*', linewidth=1, label='真实值')
plt.legend()
plt.xlabel('预测样本', size=14)
plt.ylabel('预测结果', size=14)
plt.show()


