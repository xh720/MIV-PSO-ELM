import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler

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

    def fit(self, X, Y):
        H = np.dot(X, self.weights_input_to_hidden) + self.bias_hidden
        H = 1 / (1 + np.exp(-H))  # 使用sigmoid激活函数
        self.weights_hidden_to_output = np.dot(np.linalg.pinv(H), Y)

    def predict(self, X):
        H = np.dot(X, self.weights_input_to_hidden) + self.bias_hidden
        H = 1 / (1 + np.exp(-H))
        Y_pred = np.dot(H, self.weights_hidden_to_output)
        return Y_pred

    # 粒子群优化类


class PSO:
    def __init__(self, num_particles, num_iterations, dim, omega=0.5, c1=1.0, c2=1.5):
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.dim = dim
        self.omega = omega
        self.c1 = c1
        self.c2 = c2
        self.particles = np.random.rand(num_particles, dim)
        self.velocities = np.random.rand(num_particles, dim)
        self.best_positions = np.copy(self.particles)
        self.best_scores = np.full(num_particles, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf

    def evaluate(self, elm, X_train, y_train, X_val, y_val):
        elm.weights_input_to_hidden = self.particles[0, :elm.input_size * elm.hidden_size].reshape(elm.input_size,
                                                                                                   elm.hidden_size)
        elm.bias_hidden = self.particles[0, elm.input_size * elm.hidden_size:].reshape(1, elm.hidden_size)
        elm.fit(X_train, y_train)
        y_pred = elm.predict(X_val)
        score = mean_squared_error(y_val, y_pred)
        return score

    def update_velocity_and_position(self, index):
        r1 = np.random.rand(self.dim)
        r2 = np.random.rand(self.dim)
        velocity_cognitive = self.c1 * r1 * (self.best_positions[index] - self.particles[index])
        velocity_social = self.c2 * r2 * (self.global_best_position - self.particles[index])
        self.velocities[index] = self.omega * self.velocities[index] + velocity_cognitive + velocity_social
        self.particles[index] += self.velocities[index]

    def optimize(self, elm, X_train, y_train, X_val, y_val):
        self.best_scores_history = []  # 用于记录每一代的最佳适应度值
        for _ in range(self.num_iterations):
            for i in range(self.num_particles):
                score = self.evaluate(elm, X_train, y_train, X_val, y_val)
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
        elm.weights_input_to_hidden = self.global_best_position[:elm.input_size * elm.hidden_size].reshape(
            elm.input_size, elm.hidden_size)
        elm.bias_hidden = self.global_best_position[elm.input_size * elm.hidden_size:].reshape(1, elm.hidden_size)

    def plot_optimization_process(self):
        plt.plot(self.best_scores_history, label='Best Score Over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('MSE')
        plt.title('PSO Optimization Process')
        plt.legend()
        plt.show()

# def calculate_miv(elm, X_train, y_train, model, X, y):
#     n_features = X_train.shape[1]
#     mivs = np.zeros(n_features)
#
#     # 计算每个特征对输出的影响
#     for i in range(n_features):
#         X_copy = X.copy()
#         X_copy[:, i] = np.random.normal(0, 1, X_copy.shape[0])  # 打乱当前特征
#         y_pred_orig = model.predict(X)
#         y_pred_pert = model.predict(X_copy)
#         mivs[i] = np.mean((y_pred_orig - y_pred_pert) ** 2)  # 计算均方误差
#
#     return mivs




    # 加载数据
data = pd.read_excel('F:\Python\python_program\pythonProject\A\数据集.xlsx')

# 标准化处理
scaler = StandardScaler()
X = data.iloc[1:, :-1]  # iloc表示按位置取值， :代表取所有行， :-1表示取除了最后一列的所有列
# 将数据标准化处理
X = scaler.fit_transform(X)
# 输出值不作处理
y = data.iloc[1:, -1]  # -1表示取最后一列
# 随机划分数据集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)
y_train = y_train.values.reshape(-1, 1)
print("训练集的特征值：\n", X_train, X_train.shape)

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
pso.optimize(elm, X_train, y_train, X_val, y_val)



# 绘制PSO迭代过程图
pso.plot_optimization_process()

# 使用优化后的权重和阈值进行预测
y_pred = elm.predict(X_val)



mse = mean_squared_error(y_val, y_pred)
print(f"Optimized MSE: {mse}")
y_pred = y_pred.ravel()
rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
print(f"RMSE:{rmse}")

R2 = r2_score(y_val, y_pred)
print(f"R2:{R2}")


plt.figure(1)
plt.title('测试集预测结果对比')
plt.plot(range(len(y_pred)), y_pred, 'b-o', linewidth=1, label='预测值')
plt.plot(range(len(y_val)), y_val, 'r-*', linewidth=1, label='真实值')
plt.legend()
plt.xlabel('预测样本', size=14)
plt.ylabel('预测结果', size=14)
plt.show()


